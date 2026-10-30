// PhiState MoE CPU inference adapter using the RFTMW dense FP32 kernel.
// No LibTorch, Python, autograd, or optimizer is linked into this executable.
#include "rftmw_inference.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

using Vec = std::vector<float>;
static void require(bool ok, const std::string& why) {
    if (!ok) throw std::runtime_error(why);
}
struct Reader {
    std::ifstream file;
    uint64_t remaining;
    explicit Reader(const char* path) : file(path, std::ios::binary | std::ios::ate) {
        require(bool(file), "Cannot open input");
        remaining = static_cast<uint64_t>(file.tellg()); file.seekg(0);
    }
    void read(void* out, uint64_t bytes) {
        require(bytes <= remaining, "Truncated input");
        file.read(static_cast<char*>(out), static_cast<std::streamsize>(bytes));
        require(bool(file), "Read failed"); remaining -= bytes;
    }
    int32_t integer() { int32_t v; read(&v, 4); return v; }
    Vec vector(size_t n) {
        require(n <= remaining / 4, "Tensor exceeds remaining file bytes");
        Vec v(n); read(v.data(), n * 4);
        for (float f : v) require(std::isfinite(f), "Nonfinite weight");
        return v;
    }
};
static Vec linear(const Vec& w, const Vec& x, int rows) {
    require(w.size() == size_t(rows) * x.size(), "Matrix shape mismatch");
    Vec out(rows);
    rftmw::inference::dense_sgemv_f32(w.data(), x.data(), out.data(), rows, int(x.size()));
    return out;
}
static Vec norm(const Vec& x, const Vec& weight) {
    float sum = 0; for (float v : x) sum += v * v;
    const float scale = 1.0f / std::sqrt(sum / x.size() + 1e-5f);
    Vec z(x.size());
    for (size_t i=0; i<x.size(); ++i) z[i] = x[i] * scale * weight[i];
    return z;
}
static Vec softmax(const Vec& x) {
    float largest = *std::max_element(x.begin(), x.end()), sum = 0;
    Vec p(x.size());
    for (size_t i=0; i<x.size(); ++i) { p[i]=std::exp(x[i]-largest); sum+=p[i]; }
    for (auto& v:p) v/=sum;
    return p;
}
static float sigmoid(float x) { return x >= 0 ? 1/(1+std::exp(-x)) : std::exp(x)/(1+std::exp(x)); }
struct Expert {
    Vec fc, out; int mid;
    Expert(Reader& r, int d, int ratio) : fc(r.vector(size_t(2)*d*d*ratio)),
        out(r.vector(size_t(d)*d*ratio)), mid(d*ratio) {}
    Vec operator()(const Vec& x) const {
        Vec v=linear(fc,x,2*mid), gate(mid);
        for(int i=0;i<mid;++i) gate[i]=v[i]*(v[mid+i]*sigmoid(v[mid+i]));
        return linear(out,gate,int(x.size()));
    }
};
struct Config { int d,layers,heads,window,group,vocab,experts,topk,ratio,shared,memory; };
struct Block {
    Config c;
    Vec n1,n2,qkv,proj,bias,ip,ig,op,ar,ai,cr,ci,router,n3,mixer;
    std::vector<Expert> experts,shared;
    std::vector<Vec> keys,values;
    Vec hr,hi;
    Block(Reader& r, Config cfg) : c(cfg) {
        const int d=c.d;
        n1=r.vector(d); n2=r.vector(d); qkv=r.vector(size_t(3)*d*d);
        proj=r.vector(size_t(d)*d); bias=r.vector(size_t(c.heads)*c.window);
        if(c.memory) {
            ip=r.vector(size_t(d)*d); ig=r.vector(size_t(d)*d); op=r.vector(size_t(d)*d);
            ar=r.vector(d); ai=r.vector(d); cr=r.vector(d); ci=r.vector(d);
            hr.resize(d); hi.resize(d);
        }
        router=r.vector(size_t(c.experts)*d);
        for(int e=0;e<c.experts;++e) experts.emplace_back(r,d,c.ratio);
        if(c.shared) shared.emplace_back(r,d,c.shared);
        if(c.memory) { n3=r.vector(d); mixer=r.vector(size_t(c.group)*c.group); }
    }
    Vec step(Vec x) {
        const int d=c.d, hd=d/c.heads;
        Vec z=norm(x,n1), q=linear(qkv,z,3*d);
        keys.emplace_back(q.begin()+d,q.begin()+2*d);
        values.emplace_back(q.begin()+2*d,q.end());
        if(int(keys.size())>c.window) {keys.erase(keys.begin());values.erase(values.begin());}
        const int n=int(keys.size()); Vec at(d,0);
        for(int h=0;h<c.heads;++h) {
            Vec score(n);
            for(int t=0;t<n;++t) {
                float dot=0;
                for(int j=0;j<hd;++j) dot+=q[h*hd+j]*keys[t][h*hd+j];
                score[t]=dot/std::sqrt(float(hd))+bias[h*c.window+c.window-n+t];
            }
            Vec p=softmax(score);
            for(int j=0;j<hd;++j) for(int t=0;t<n;++t) at[h*hd+j]+=p[t]*values[t][h*hd+j];
        }
        at=linear(proj,at,d);
        if(c.memory) {
            Vec u=linear(ip,z,d),g=linear(ig,z,d),y(d);
            for(int j=0;j<d;++j) {
                const float nr=ar[j]*hr[j]-ai[j]*hi[j]+u[j]*sigmoid(g[j]);
                hi[j]=ar[j]*hi[j]+ai[j]*hr[j]; hr[j]=nr;
                y[j]=hr[j]*cr[j]-hi[j]*ci[j];
            }
            y=linear(op,y,d);
            for(int j=0;j<d;++j) at[j]=(at[j]+y[j])*float(1/std::sqrt(2.0));
        }
        for(int j=0;j<d;++j) x[j]+=0.7f*at[j];
        z=norm(x,n2); Vec scores=linear(router,z,c.experts);
        std::vector<int> order(c.experts); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[a]>scores[b];});
        Vec selected;
        for(int k=0;k<c.topk;++k) selected.push_back(scores[order[k]]);
        Vec gate=softmax(selected), routed(d,0);
        if(c.topk==1) gate[0]=softmax(scores)[order[0]];
        // Accumulate in expert-index order, matching SparseMoE.forward.
        for(int e=0;e<c.experts;++e) for(int k=0;k<c.topk;++k) if(order[k]==e) {
            Vec y=experts[e](z);
            for(int j=0;j<d;++j) routed[j]+=y[j]*gate[k];
        }
        if(c.shared) { Vec y=shared[0](z); for(int j=0;j<d;++j) routed[j]+=y[j]; }
        for(int j=0;j<d;++j) x[j]+=0.5f*routed[j];
        if(c.memory) {
            z=norm(x,n3);
            for(int i=0;i<d;i+=c.group) {
                Vec group(z.begin()+i,z.begin()+i+c.group), y=linear(mixer,group,c.group);
                for(int j=0;j<c.group;++j) x[i+j]+=0.5f*(y[j]-z[i+j]);
            }
        }
        return x;
    }
};
struct Model {
    Config c; Vec embedding,final_norm; std::vector<Block> blocks;
    explicit Model(const char* path) {
        uint16_t endian=1; require(*reinterpret_cast<char*>(&endian)==1,"Little endian host required");
        Reader r(path); char magic[8]; r.read(magic,8);
        require(std::memcmp(magic,"PHIMOE1\n",8)==0,"Wrong native model format");
        c.d=r.integer(); c.layers=r.integer(); c.heads=r.integer(); c.window=r.integer();
        c.group=r.integer(); c.vocab=r.integer(); c.experts=r.integer(); c.topk=r.integer();
        c.ratio=r.integer(); c.shared=r.integer(); c.memory=r.integer();
        require(c.d>0 && c.d<=4096 && c.layers>0 && c.layers<=128 && c.heads>0 &&
          c.heads<=c.d && c.d%c.heads==0 && c.group>0 && c.d%c.group==0 &&
          c.window>0 && c.window<=65536 && c.vocab>=256 && c.vocab<=65535 &&
          c.experts>=2 && c.experts<=256 && c.topk>=1 && c.topk<=c.experts &&
          c.ratio>0 && c.ratio<=32 && c.shared>=0 && c.shared<=32 &&
          (c.memory==0 || c.memory==1),"Invalid native model dimensions");
        embedding=r.vector(size_t(c.vocab)*c.d); final_norm=r.vector(c.d);
        for(int i=0;i<c.layers;++i) blocks.emplace_back(r,c);
        require(r.remaining==0,"Unexpected trailing model bytes");
    }
    Vec step(int token) {
        require(token>=0 && token<c.vocab,"Token ID out of range");
        Vec x(embedding.begin()+size_t(token)*c.d,embedding.begin()+size_t(token+1)*c.d);
        for(auto& b:blocks) x=b.step(std::move(x));
        Vec logits=linear(embedding,norm(x,final_norm),c.vocab);
        for(float v:logits) require(std::isfinite(v),"Nonfinite native logits");
        return logits;
    }
};
int main(int argc,char** argv) {
    try {
        require(argc==6,"Usage: phistate_moe_cli MODEL TOKENS_U32 LOGITS_F32 NEW_TOKENS EOS_ID");
        Model model(argv[1]); Reader tokens(argv[2]);
        require(tokens.remaining>0 && tokens.remaining%4==0 && tokens.remaining<=4000000,"Invalid token file");
        std::vector<int32_t> ids(tokens.remaining/4); tokens.read(ids.data(),ids.size()*4);
        const int count=std::stoi(argv[4]),eos=std::stoi(argv[5]);
        require(count>=0 && count<=4096,"Invalid generation length");
        std::ofstream logits_file(argv[3],std::ios::binary); require(bool(logits_file),"Cannot create logits output");
        Vec logits;
        const auto start=std::chrono::steady_clock::now();
        for(int token:ids) { logits=model.step(token); logits_file.write(reinterpret_cast<char*>(logits.data()),logits.size()*4); }
        require(bool(logits_file),"Logits write failed");
        std::vector<int> generated;
        for(int i=0;i<count;++i) {
            int token=int(std::max_element(logits.begin(),logits.end())-logits.begin());
            if(token==eos) break;
            generated.push_back(token);
            if(i+1<count) logits=model.step(token);
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout << "{\"backend\":\"RFTMW C++ FP32 CPU\",\"seconds\":" << seconds
                  << ",\"prompt_tokens\":" << ids.size() << ",\"generated_ids\":[";
        for(size_t i=0;i<generated.size();++i) {if(i) std::cout<<',';std::cout<<generated[i];}
        std::cout << "]}\n";
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
