#!/usr/bin/env python3
"""Wan dense layers -> QuantoniumOS's AVX2/FMA assembly + native thread pool.

Inference only, CPU FP32. Attention softmax/SDPA, norms, activations and VAE
remain PyTorch. No RFT approximation or promise of faster/better generation.
This is a local adapter around the user's existing staged CPU renderer.
"""
from __future__ import annotations
import argparse, ast, ctypes, functools, hashlib, importlib.util, json, os
from pathlib import Path
import platform, shutil, subprocess, sys, tempfile, time, types

CPP = r'''
#define RFTMW_ENABLE_NATIVE_GEMV_ASM 1
#include "rftmw_inference.hpp"
#include <climits>
#include <cstddef>
extern "C" void rftmw_dot_f32_asm(const float*, const float*, float*, int);
extern "C" int qos_cpu_supported() {
#if defined(__x86_64__) && defined(__GNUC__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return 0;
#endif
}
// X[M,K], W[N,K], optional b[N] -> Y[M,N]. This is NOT a BLAS fallback.
extern "C" int qos_linear(const float* x, const float* w, const float* b,
                          float* y, int m, int n, int k) noexcept {
    if(!qos_cpu_supported()) return 1;
    if(!x || !w || !y || m<=0 || n<=0 || k<=0) return 2;
    constexpr int TILE=16;
    const long long tiles=(static_cast<long long>(m)+TILE-1)/TILE;
    if(tiles*n>INT_MAX) return 3;
    try {
        const int jobs=static_cast<int>(tiles*n);
        const size_t work=static_cast<size_t>(m)*n*k*sizeof(float);
        rftmw::inference::parallel_for(jobs, work, [&](int a,int z){
            for(int j=a;j<z;++j){
                const int row=j%n, begin=(j/n)*TILE;
                const float* wr=w+static_cast<size_t>(row)*k;
                const int end=std::min(begin+TILE,m);
                for(int t=begin;t<end;++t){
                    float* dst=y+static_cast<size_t>(t)*n+row;
                    rftmw_dot_f32_asm(wr,x+static_cast<size_t>(t)*k,dst,k);
                    if(b) *dst+=b[row];
                }
            }
        });
    } catch(...) {return 4;}
    return 0;
}
'''


def require(ok, message):
    if not ok: raise RuntimeError(message)


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(p, value):
    p=Path(p); tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n');tmp.replace(p)


def build(native: Path, out: Path, gas_test_only=False):
    """Compile the actual repository kernel; never replace it with torch/BLAS."""
    require(platform.machine().lower() in ('x86_64','amd64'), 'This kernel needs x86-64.')
    flags=Path('/proc/cpuinfo').read_text()
    require('avx2' in flags and 'fma' in flags, 'AVX2 and FMA are required; refusing a silent fallback.')
    asm=native/'rftmw_sgemv_f32.asm'; header=native/'rftmw_inference.hpp'
    require(asm.is_file() and header.is_file(), f'Native sources missing in {native}')
    out.mkdir(parents=True,exist_ok=True)
    src=out/'bridge.cpp';src.write_text(CPP)
    obj=out/'dot.o';lib=out/'libqos_wan_linear.so'
    compiler=shutil.which('g++') or shutil.which('c++')
    require(compiler is not None, 'C++ compiler missing.')
    assembler='nasm'
    if gas_test_only:
        # Local validation only: preserve the upstream instruction body,
        # converting NASM directives to GAS Intel syntax. Production uses NASM.
        lines=['.intel_syntax noprefix']
        for raw in asm.read_text().splitlines():
            s=raw.split(';',1)[0].strip()
            if not s or s=='default rel' or s.startswith('%endif'): continue
            if s.startswith('%ifnidn __OUTPUT_FORMAT__, win64'): continue
            require(not s.startswith('%'), 'Unexpected assembler preprocessor directive')
            if s.startswith('section '): s='.'+s.split()[1].lstrip('.')
            elif s.startswith('global '): s='.globl '+s.split()[1]
            elif s.startswith('align '): s='.balign '+s.split()[1]
            lines.append(s)
        lines.append('.section .note.GNU-stack,"",@progbits')
        gas=out/'dot.S';gas.write_text('\n'.join(lines)+'\n')
        subprocess.run([compiler,'-c','-fPIC',str(gas),'-o',str(obj)],check=True)
        assembler='gas-syntax-translation-test-only'
    else:
        require(shutil.which('nasm') is not None,'NASM missing; native build did not run.')
        subprocess.run(['nasm','-f','elf64',str(asm),'-o',str(obj)],check=True)
    subprocess.run([compiler,'-std=c++17','-O3','-fPIC','-shared','-pthread',
                    '-I',str(native),str(src),str(obj),'-Wl,-z,noexecstack',
                    '-o',str(lib)],check=True)
    info=dict(kernel='rftmw_dot_f32_asm', scheduler='rftmw::inference::parallel_for',
              assembler=assembler, asm_sha256=digest(asm), header_sha256=digest(header),
              wrapper_sha256=digest(src), library_sha256=digest(lib))
    write_json(out/'build.json',info)
    return lib,info


class Engine:
    def __init__(self, library: Path):
        import torch
        self.torch=torch;self.lib=ctypes.CDLL(str(library))
        self.lib.qos_cpu_supported.restype=ctypes.c_int
        require(self.lib.qos_cpu_supported(), 'Native CPU feature check failed.')
        self.fn=self.lib.qos_linear
        self.fn.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3
        self.fn.restype=ctypes.c_int
        self.rows={};self.checks={};self.installed=[];self.enabled=True

    def raw(self,x,w,b=None):
        torch=self.torch
        require(x.ndim>=1 and w.ndim==2 and x.shape[-1]==w.shape[1], 'Linear shape mismatch')
        for v in (x,w,b):
            if v is not None:
                require(v.device.type=='cpu' and v.dtype==torch.float32,
                        'Native linear requires CPU FP32; no silent fallback.')
        require(not torch.is_grad_enabled(), 'Native bridge is inference-only, not autograd training.')
        x=x.contiguous();w=w.contiguous();b=None if b is None else b.contiguous()
        m=x.numel()//w.shape[1];n,k=w.shape
        out=torch.empty((*x.shape[:-1],n),dtype=torch.float32)
        if m:
            require(max(m,n,k)<2**31,'Native dimension overflow')
            status=self.fn(x.data_ptr(),w.data_ptr(),0 if b is None else b.data_ptr(),
                           out.data_ptr(),m,n,k)
            require(status==0,f'Native kernel failed, status={status}')
        return out

    def compare(self, actual, reference, name):
        torch=self.torch
        require(torch.isfinite(actual).all().item() and torch.isfinite(reference).all().item(),
                f'Nonfinite output at {name}')
        delta=actual-reference
        l2=float(torch.linalg.vector_norm(delta)/(torch.linalg.vector_norm(reference)+1e-12))
        mx=float(delta.abs().max()) if delta.numel() else 0.
        scale=max(1.,float(reference.abs().max())) if reference.numel() else 1.
        require(l2<=3e-5 and mx<=3e-5*scale,
                f'Native parity FAILED at {name}: relative_l2={l2}, max_abs={mx}')
        return dict(relative_l2=l2,max_abs=mx,criterion='l2<=3e-5 and max_abs<=3e-5*max(1,ref_max)')

    def forward(self,module,x,name):
        torch=self.torch
        if not self.enabled:
            return torch.nn.functional.linear(x,module.weight,module.bias)
        if name not in self.checks:
            sample=x.reshape(-1,x.shape[-1])[:2].contiguous()
            reference=torch.nn.functional.linear(sample,module.weight,module.bias)
            self.checks[name]=self.compare(self.raw(sample,module.weight,module.bias),reference,name)
        t=time.perf_counter();y=self.raw(x,module.weight,module.bias)
        row=self.rows.setdefault(name,dict(calls=0,dot_calls=0,seconds=0.,in_features=module.in_features,out_features=module.out_features))
        row['calls']+=1;row['dot_calls']+=(x.numel()//module.in_features)*module.out_features
        row['seconds']+=time.perf_counter()-t
        return y

    def install(self,model):
        for name,module in model.named_modules():
            if isinstance(module,self.torch.nn.Linear):
                require(type(module) is self.torch.nn.Linear,
                        f'Unexpected adapted Linear at {name}; LoRA subclasses need separate integration.')
                def forward(m,x,_name=name): return self.forward(m,x,_name)
                module.forward=types.MethodType(forward,module)
                self.installed.append(name)
        require(self.installed,'No Wan linear modules were installed')
        print(f'NATIVE WIRED: {len(self.installed)} Wan Linear modules -> rftmw_dot_f32_asm',flush=True)

    def summary(self):
        return dict(backend='QuantoniumOS AVX2/FMA + native thread pool',
                    linear_calls=sum(x['calls'] for x in self.rows.values()),
                    asm_dot_calls=sum(x['dot_calls'] for x in self.rows.values()),
                    installed_modules=len(self.installed), executed_modules=len(self.rows),
                    module_calls=self.rows,parity_checks=self.checks,
                    remaining_torch=['SDPA/softmax','normalization','activations','patch convolution','text encoder','VAE'],
                    rft_in_neural_math=False,lora_applied=False)

    def smoke(self):
        torch=self.torch;g=torch.Generator().manual_seed(808);results=[]
        with torch.inference_mode():
            for m,n,k in [(1,9,1),(3,13,7),(17,35,8),(5,31,31),(2,45,32),
                          (4,33,33),(16,64,128),(2,1536,1536),(2,64,8960)]:
                x=torch.randn(m,k,generator=g);w=torch.randn(n,k,generator=g)/(k**.5)
                b=torch.randn(n,generator=g)
                results.append(dict(shape=[m,n,k],**self.compare(self.raw(x,w,b),
                                 torch.nn.functional.linear(x,w,b),f'smoke {m,n,k}')))
        print(f'NATIVE PARITY: {len(results)} matrix cases passed',flush=True)
        return results


def load_module(path:Path):
    spec=importlib.util.spec_from_file_location('qos_existing_renderer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path('/workspaces/quantonium-wan-cpu-1d7a4b6ce30d'))
    p.add_argument('--renderer',type=Path,default=Path('/home/codespace/.cache/qos-wan-render-dwOmKC.py'))
    p.add_argument('--cache',type=Path,default=Path(f'/tmp/qos-wan-{os.getuid()}/.cache'))
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--steps',type=int,default=30)
    p.add_argument('--size',type=int,default=256)
    p.add_argument('--frames',type=int,default=17)
    p.add_argument('--self-test-only',action='store_true')
    p.add_argument('--native-dir',type=Path)
    p.add_argument('--gas-test-only',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    require(1<=args.threads<=max(1,len(os.sched_getaffinity(0))),'Invalid thread count')
    require(1<=args.steps<=50 and 128<=args.size<=512 and args.size%16==0,'Invalid size/steps')
    require(5<=args.frames<=49 and (args.frames-1)%4==0,'Invalid frame count')
    require(not args.gas_test_only or args.self_test_only,'GAS translation is only for local kernel tests.')
    work=args.work.resolve();native=args.native_dir or work/'runtime/src/rftmw_native'
    os.environ.update(RFTMW_THREADS=str(args.threads),OMP_NUM_THREADS=str(args.threads),
                      MKL_NUM_THREADS=str(args.threads),OPENBLAS_NUM_THREADS=str(args.threads),
                      QOS_STEPS=str(args.steps),QOS_SIZE=str(args.size),QOS_FRAMES=str(args.frames),
                      QOS_MODE='staged')
    result_root=work/'video_cpu_results';result_root.mkdir(parents=True,exist_ok=True)
    out=Path(tempfile.mkdtemp(prefix='native-launch-',dir=result_root))
    print('NATIVE LAUNCH EVIDENCE:',out,flush=True)
    library,build_info=build(native,out,args.gas_test_only)
    import torch
    torch.set_num_threads(args.threads)
    engine=Engine(library);smoke=engine.smoke()
    write_json(out/'kernel-tests.json',dict(build=build_info,tests=smoke))
    if args.self_test_only: return
    require(args.renderer.is_file(),f'Existing renderer missing: {args.renderer}')
    # Preserve its original cached prompts/weights. No resets or deletions.
    cache=args.cache.resolve()
    require(cache.is_dir(),f'Scratch cache missing: {cache}; original model files untouched.')
    require(shutil.disk_usage(cache).free>=10*(1<<30),'Need 10 GiB scratch headroom.')
    hf=cache/'huggingface';hf.mkdir(exist_ok=True)
    os.environ['HOME']=str(cache.parent)  # downloader also uses Path.home()
    for key in ('TRANSFORMERS_CACHE','DIFFUSERS_CACHE','HUGGINGFACE_HUB_CACHE'):
        os.environ.pop(key,None)
    os.environ.update(HF_HOME=str(hf),HF_HUB_CACHE=str(hf/'hub'),
                      HF_HUB_OFFLINE='0',TRANSFORMERS_OFFLINE='0')
    import diffusers
    require(diffusers.__version__=='0.35.2','Use the existing Diffusers 0.35.2 environment.')
    from diffusers import WanTransformer3DModel, WanPipeline
    renderer=load_module(args.renderer)
    renderer.WORK=work;renderer.CACHE=cache
    renderer.BENCH=cache/'qos-wan-scale-ASlSXB/benchmark_8_blocks.py'
    renderer.OLD=cache/'qos-wan-block0-lzg183le'
    require(renderer.BENCH.is_file() and (renderer.OLD/'weights.json').is_file(),
            'Existing benchmark / block-0 cache missing; no replacement weights invented.')
    renderer.STEPS=args.steps;renderer.SIZE=args.size;renderer.FRAMES=args.frames;renderer.MODE='staged'
    # Retain the known renderer. Insert native computation when its model is created.
    original_factory=WanTransformer3DModel.from_config
    @classmethod
    def factory(cls,*a,**kw):
        model=original_factory(*a,**kw)
        engine.install(model)
        original_forward=model.forward
        checked=False
        def verified_forward(m,*fa,**fk):
            nonlocal checked
            if checked:
                return original_forward(*fa,**fk)
            print('FIRST DENOISER PASS: checking native output against PyTorch on identical inputs.',flush=True)
            try:
                engine.enabled=False
                reference=original_forward(*fa,**fk)
            finally:
                engine.enabled=True
            actual=original_forward(*fa,**fk)
            a_tensor=actual[0] if isinstance(actual,tuple) else actual.sample
            r_tensor=reference[0] if isinstance(reference,tuple) else reference.sample
            delta=a_tensor-r_tensor
            l2=float(torch.linalg.vector_norm(delta)/(torch.linalg.vector_norm(r_tensor)+1e-12))
            mx=float(delta.abs().max());scale=max(1.,float(r_tensor.abs().max()))
            require(torch.isfinite(a_tensor).all().item() and l2<=3e-4 and mx<=3e-4*scale,
                    f'Full-transformer native parity failed: l2={l2}, max_abs={mx}; no full render accepted.')
            write_json(out/'first-transformer-parity.json',dict(relative_l2=l2,max_abs=mx,
                       scope='same extracted weights, same first latent and conditioning; not an independent source-weight audit'))
            print(f'FULL TRANSFORMER NATIVE PARITY: relative_l2={l2:.3g}, max_abs={mx:.3g}',flush=True)
            checked=True
            return actual
        model.forward=types.MethodType(verified_forward,model)
        return model
    WanTransformer3DModel.from_config=factory
    # Native evidence is recorded with every denoising step and in the run report.
    original_call=WanPipeline.__call__
    @functools.wraps(original_call)
    def call(pipe,*a,**kw):
        callback=kw.get('callback_on_step_end')
        def step(pipe,index,timestep,values):
            values=callback(pipe,index,timestep,values) if callback else values
            data=engine.summary();data['completed_step']=index+1
            write_json(out/'native-calls.json',data)
            print(f'NATIVE STEP {index+1}: Linear calls={data["linear_calls"]}; ASM dots={data["asm_dot_calls"]}',flush=True)
            return values
        kw['callback_on_step_end']=step
        return original_call(pipe,*a,**kw)
    WanPipeline.__call__=call
    original_write=renderer.write_json
    def save(path,value):
        if Path(path).name=='report.json':
            value=dict(value,native_kernel=engine.summary(),native_build=build_info,
                       native_launcher=str(Path(__file__).resolve()),native_launch_evidence=str(out))
            (out/'render_report_path.txt').write_text(str(path)+'\n')
        return original_write(path,value)
    renderer.write_json=save
    status='FAILED_OR_INTERRUPTED'
    try:
        print('RENDER: staged weights + actual native Wan Linear computation; no quality claim yet.',flush=True)
        renderer.main()
        need_names=('attn1.to_q','attn1.to_k','attn1.to_v','ffn.net.','proj_out')
        require(all(any(n in k for k in engine.rows) for n in need_names),
                'Render did not exercise required native operations; refusing a native-render success.')
        status='RENDER_COMPLETED_NATIVE_CALLS_VERIFIED_VISUAL_REVIEW_REQUIRED'
    finally:
        write_json(out/'native-calls.json',dict(status=status,build=build_info,**engine.summary()))
        print('NATIVE REPORT:',out/'native-calls.json',flush=True)
        WanTransformer3DModel.from_config=original_factory
        WanPipeline.__call__=original_call


if __name__=='__main__':main()
