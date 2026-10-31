// PhiState MoE CPU inference adapter using the RFTMW dense FP32 kernel.
// No LibTorch, Python, autograd, or optimizer is linked into this executable.
#include "rftmw_inference.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
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
struct DecodeConfig {
    double temperature=0.0, top_p=1.0, repetition_penalty=1.0;
    int top_k=0;
    uint64_t seed=0;
    void validate() const {
        require(std::isfinite(temperature) && temperature>=0, "Invalid temperature");
        require(std::isfinite(top_p) && top_p>0 && top_p<=1, "Invalid top_p");
        require(top_k>=0, "Invalid top_k");
        require(std::isfinite(repetition_penalty) && repetition_penalty>=1,
                "Invalid repetition_penalty");
    }
};
static double parse_real(const char* text) {
    size_t pos=0; double value=std::stod(text,&pos);
    require(pos==std::strlen(text), "Invalid numeric argument"); return value;
}
static int parse_int(const char* text) {
    size_t pos=0; long long value=std::stoll(text,&pos);
    require(pos==std::strlen(text) && value>=std::numeric_limits<int>::min() &&
            value<=std::numeric_limits<int>::max(), "Invalid integer argument");
    return int(value);
}
static uint64_t parse_seed(const char* text) {
    require(text[0]!='-', "Invalid seed");
    size_t pos=0; auto value=std::stoull(text,&pos);
    require(pos==std::strlen(text), "Invalid seed"); return value;
}
static int choose_token(const Vec& logits, const std::vector<bool>& seen, int eos,
                        const DecodeConfig& cfg, std::mt19937_64& rng) {
    require(!logits.empty() && seen.size()==logits.size(), "Invalid sampling input");
    std::vector<double> scores(logits.begin(),logits.end());
    for(size_t i=0;i<scores.size();++i) {
        require(std::isfinite(scores[i]), "Nonfinite sampling logits");
        // Apply the penalty once per generated ID, not to the prompt or EOS.
        if(seen[i] && int(i)!=eos)
            scores[i] = scores[i]>=0 ? scores[i]/cfg.repetition_penalty
                                    : scores[i]*cfg.repetition_penalty;
        require(std::isfinite(scores[i]), "Nonfinite penalized sampling logits");
    }
    if(cfg.temperature==0)
        return int(std::max_element(scores.begin(),scores.end())-scores.begin());
    std::vector<int> order(scores.size()); std::iota(order.begin(),order.end(),0);
    std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[a]>scores[b];});
    if(cfg.top_k>0 && size_t(cfg.top_k)<order.size()) order.resize(cfg.top_k);
    const double largest=scores[order.front()];
    std::vector<double> weights;
    double total=0;
    for(int token:order) {
        double weight=std::exp((scores[token]-largest)/cfg.temperature);
        weights.push_back(weight); total+=weight;
    }
    require(total>0 && std::isfinite(total), "Invalid sampling distribution");
    // Include the token that crosses the nucleus threshold; always keep one.
    if(cfg.top_p<1) {
        double cumulative=0; size_t keep=0;
        do { cumulative+=weights[keep++]; } while(keep<weights.size() && cumulative<cfg.top_p*total);
        order.resize(keep); weights.resize(keep); total=cumulative;
    }
    // Explicit 53-bit uniform conversion makes seeded draws independent of the
    // standard library's discrete_distribution implementation.
    const double draw=double(rng()>>11)*0x1.0p-53*total;
    double cumulative=0;
    for(size_t i=0;i<weights.size();++i) {
        cumulative+=weights[i]; if(draw<cumulative) return order[i];
    }
    return order.back();
}
int main(int argc,char** argv) {
    try {
        require(argc==6 || argc==11,"Usage: phistate_moe_cli MODEL TOKENS_U32 LOGITS_F32 NEW_TOKENS EOS_ID [TEMPERATURE TOP_P TOP_K REPETITION_PENALTY SEED]");
        DecodeConfig decode;
        if(argc==11) {
            decode.temperature=parse_real(argv[6]); decode.top_p=parse_real(argv[7]);
            decode.top_k=parse_int(argv[8]); decode.repetition_penalty=parse_real(argv[9]);
            decode.seed=parse_seed(argv[10]);
        }
        decode.validate();
        Model model(argv[1]); Reader tokens(argv[2]);
        require(tokens.remaining>0 && tokens.remaining%4==0 && tokens.remaining<=4000000,"Invalid token file");
        std::vector<int32_t> ids(tokens.remaining/4); tokens.read(ids.data(),ids.size()*4);
        const int count=parse_int(argv[4]),eos=parse_int(argv[5]);
        require(count>=0 && count<=4096,"Invalid generation length");
        require(eos>=-1 && eos<model.c.vocab,"Invalid EOS token");
        std::ofstream logits_file(argv[3],std::ios::binary); require(bool(logits_file),"Cannot create logits output");
        Vec logits;
        const auto start=std::chrono::steady_clock::now();
        for(int token:ids) { logits=model.step(token); logits_file.write(reinterpret_cast<char*>(logits.data()),logits.size()*4); }
        require(bool(logits_file),"Logits write failed");
        std::vector<int> generated;
        std::vector<bool> seen(model.c.vocab,false);
        std::mt19937_64 rng(decode.seed);
        bool ended=false;
        for(int i=0;i<count;++i) {
            int token=choose_token(logits,seen,eos,decode,rng);
            if(token==eos) { ended=true; break; }
            generated.push_back(token);
            seen[token]=true;
            if(i+1<count) logits=model.step(token);
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        std::cout << "{\"backend\":\"RFTMW C++ FP32 CPU\",\"seconds\":" << seconds
                  << ",\"prompt_tokens\":" << ids.size() << ",\"generated_ids\":[";
        for(size_t i=0;i<generated.size();++i) {if(i) std::cout<<',';std::cout<<generated[i];}
        std::cout << "],\"eos\":" << (ended ? "true" : "false")
                  << ",\"decoding_api\":1,\"temperature\":" << decode.temperature
                  << ",\"top_p\":" << decode.top_p << ",\"top_k\":" << decode.top_k
                  << ",\"repetition_penalty\":" << decode.repetition_penalty
                  << ",\"seed\":" << decode.seed << "}\n";
        return 0;
    } catch(const std::exception& e) {std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
