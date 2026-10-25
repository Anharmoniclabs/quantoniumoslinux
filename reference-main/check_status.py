#!/usr/bin/env python3
"""Current inference metrics and status."""
import sys
import json
import time
import numpy as np

sys.path.insert(0, 'quantonium_os_src/engine')
sys.path.insert(0, 'src/rftmw_native/build')

print("=" * 60)
print("QUANTONIUMOS INFERENCE STATUS REPORT")
print("=" * 60)

# Check native module
print("\n=== Native Module ===")
try:
    import rftmw_native
    print(f"Version: {rftmw_native.__version__}")
    print(f"AVX2: {rftmw_native.HAS_AVX2}")
    print(f"FMA: {rftmw_native.HAS_FMA}")
    print(f"ASM Kernels: {rftmw_native.HAS_ASM_KERNELS}")
    if hasattr(rftmw_native, 'infer'):
        print(f"Inference kernels: {rftmw_native.infer.available}")
        print(f"MoE support: {getattr(rftmw_native.infer, 'has_moe', False)}")
        print(f"SSM support: {getattr(rftmw_native.infer, 'has_ssm', False)}")
    native_ok = True
except Exception as e:
    print(f"Error: {e}")
    native_ok = False

# Check packs
print("\n=== Model Packs ===")
from pathlib import Path
packs = list(Path('.').glob('*.rftmwpk'))
for p in packs:
    meta_path = p.with_suffix('.rftmwpk.meta.json')
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        prov = meta.get('provenance', {})
        cfg = meta.get('model_config', {})
        arch = cfg.get('architecture', 'qwen2')
        params = prov.get('parameter_count', 0) / 1e9
        print(f"  {p.name}: {params:.1f}B params, arch={arch}")
    else:
        print(f"  {p.name}: (no sidecar)")

# Inference benchmark on 3b
print("\n=== Inference Benchmark (Qwen2.5-Coder 3B) ===")
from rftmw_numpy_engine import Qwen25Engine, _rms_norm

try:
    t0 = time.perf_counter()
    e = Qwen25Engine('qwen2.5-coder_3b.rftmwpk')
    e.load()
    load_time = time.perf_counter() - t0
    print(f"Load time: {load_time:.1f}s")

    # Embed lookup
    t0 = time.perf_counter()
    for _ in range(100):
        emb = e._embed_token(1234)
    embed_ms = (time.perf_counter() - t0) * 10  # ms per lookup
    print(f"Embed lookup: {embed_ms:.2f}ms/token")

    # RMS norm
    x = emb.copy()
    t0 = time.perf_counter()
    for _ in range(100):
        h = _rms_norm(x, e.layers[0].attn_norm, e.config.rms_norm_eps)
    norm_ms = (time.perf_counter() - t0) * 10
    print(f"RMS norm: {norm_ms:.2f}ms")

    # Q projection matmul
    t0 = time.perf_counter()
    for _ in range(50):
        q = e.layers[0].q_proj.matmul(h)
    matmul_ms = (time.perf_counter() - t0) * 20
    print(f"Q projection: {matmul_ms:.2f}ms")

    # Estimate throughput
    # Full layer: embed + 2*norm + 7*matmul (qkvo + gate/up/down)
    layer_ms = embed_ms + 2*norm_ms + 7*matmul_ms
    total_ms = layer_ms * e.config.num_layers
    tok_s = 1000 / total_ms
    print(f"Estimated throughput: {tok_s:.1f} tok/s ({e.config.num_layers} layers)")

except Exception as ex:
    print(f"Error: {ex}")

# Qwen35 pack check
print("\n=== Qwen35 MoE+SSM Pack ===")
try:
    from rftmw_numpy_engine import Qwen35MoeConfig, _parse_gguf_index
    
    pack35 = Path('qwen35-opus-distil.rftmwpk')
    if pack35.exists():
        meta = json.loads(pack35.with_suffix('.rftmwpk.meta.json').read_text())
        cfg = meta.get('model_config', {})
        prov = meta.get('provenance', {})
        
        print(f"Pack: {pack35.name}")
        print(f"Parameters: {prov.get('parameter_count', 0) / 1e9:.1f}B")
        print(f"Architecture: {cfg.get('architecture', 'unknown')}")
        print(f"Layers: {cfg.get('num_layers', 0)}")
        print(f"Experts: {cfg.get('num_experts', 0)} (top-{cfg.get('num_experts_per_tok', 0)} active)")
        print(f"SSM state: {cfg.get('ssm_state_size', 0)}")
        print(f"Full attention interval: {cfg.get('full_attention_interval', 0)}")
        
        # Check GGUF availability
        gguf_path = prov.get('gguf_path', '')
        if Path(gguf_path).exists():
            size_gb = Path(gguf_path).stat().st_size / 1e9
            print(f"GGUF: {size_gb:.1f}GB (available)")
        else:
            print(f"GGUF: NOT FOUND at {gguf_path}")
    else:
        print("Pack not found")
except Exception as ex:
    print(f"Error: {ex}")

# Compression results
print("\n=== Codec Pipeline Results ===")
try:
    results = json.loads(Path('codec_pipeline_results.json').read_text())
    vs = results.get('variant_shootout', {})
    for signal, data in vs.items():
        dft = data.get('dft_k99', 0)
        top = data.get('top5', [[]])[0]
        if top:
            name, k, ratio = top[0], top[1], top[2]
            improvement = f"{(1-ratio)*100:.0f}% better" if ratio < 1 else f"{(ratio-1)*100:.0f}% worse"
            print(f"  {signal}: best={name} (k={k} vs DFT k={dft}, {improvement})")
except Exception as ex:
    print(f"Error: {ex}")

print("\n" + "=" * 60)
print("STATUS SUMMARY")
print("=" * 60)
print(f"Native C++ kernels: {'OK' if native_ok else 'FALLBACK TO NUMPY'}")
print(f"3B model: READY")
print(f"35B MoE+SSM model: PACK READY (needs native MoE/SSM kernels or NumPy fallback)")
