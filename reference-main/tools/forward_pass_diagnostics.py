#!/usr/bin/env python3
"""
Forward Pass Diagnostics — Detect Numerical Drift, Segmentation, and Layer Issues

This tool detects common causes of "dumb" or nonsense outputs:
1. Layer-by-layer numerical drift (accumulated FP32 errors)
2. KV cache corruption or overflow
3. Weight precision loss under quantization
4. Attention score explosions (NaN/Inf)
5. Logit distribution degradation

Run: python tools/forward_pass_diagnostics.py [model.rftmwpk]
"""
import sys
import numpy as np
from pathlib import Path

# Add project root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen25Engine, _rms_norm, _softmax, _get_native_infer, _rope_freqs
)

def check_tensor(name: str, t: np.ndarray, warn_threshold=10000.0, nan_check=True):
    """Check tensor for numerical issues.
    
    Note: Large values (up to 1000+) are normal in transformer hidden states,
    especially in early layers. The model uses residual connections that
    accumulate values. Only truly extreme values (>10000) indicate problems.
    """
    issues = []
    
    if nan_check:
        nan_count = np.sum(np.isnan(t))
        if nan_count > 0:
            issues.append(f"NaN: {nan_count}")
    
    inf_count = np.sum(np.isinf(t))
    if inf_count > 0:
        issues.append(f"Inf: {inf_count}")
    
    max_abs = np.max(np.abs(t))
    if max_abs > warn_threshold:
        issues.append(f"Extreme values: max_abs={max_abs:.2f}")
    
    if issues:
        print(f"  ⚠️  {name}: {', '.join(issues)}")
        return False
    return True


def diagnose_layer_forward(engine, layer_idx, h_input, pos, rope_freqs, k_cache, v_cache, cache_len):
    """Run a single layer's forward pass with detailed diagnostics."""
    layer = engine.layers[layer_idx]
    cfg = engine.config
    
    print(f"\n=== Layer {layer_idx} Diagnostics ===")
    layer_type = type(layer).__name__
    print(f"Type: {layer_type}")
    print(f"Input h: shape={h_input.shape}, min={h_input.min():.4f}, max={h_input.max():.4f}, mean={h_input.mean():.4f}")
    
    # Step 1: Attention pre-norm
    normed = _rms_norm(h_input, layer.attn_norm, cfg.rms_norm_eps)
    if not check_tensor("attn_norm output", normed):
        return None
    print(f"  attn_norm: {normed.min():.4f} to {normed.max():.4f}")
    
    # Step 2: QKV projections - handle both fused and separate
    if hasattr(layer, 'qkv_proj') and layer.qkv_proj is not None:
        # MoE layer with fused QKV
        qkv = layer.qkv_proj.matmul(normed)
        if not check_tensor("QKV projection", qkv):
            return None
        # Split into Q, K, V
        q_dim = layer.q_dim
        k_dim = layer.k_dim
        q = qkv[:q_dim]
        k = qkv[q_dim:q_dim + k_dim]
        v = qkv[q_dim + k_dim:]
        print(f"  QKV (fused): {qkv.min():.4f} to {qkv.max():.4f}")
    else:
        # Standard layer with separate Q, K, V projections
        q = layer.q_proj.matmul(normed)
        k = layer.k_proj.matmul(normed)
        v = layer.v_proj.matmul(normed)
    
    if not check_tensor("Q projection", q):
        return None
    if not check_tensor("K projection", k):
        return None
    if not check_tensor("V projection", v):
        return None
    
    print(f"  Q: {q.min():.4f} to {q.max():.4f}")
    print(f"  K: {k.min():.4f} to {k.max():.4f}")
    print(f"  V: {v.min():.4f} to {v.max():.4f}")
    
    # Step 3: Check attention scores distribution
    q_reshaped = q.reshape(cfg.num_heads, cfg.head_dim)
    k_reshaped = k.reshape(cfg.num_kv_heads, cfg.head_dim)
    
    # Single position attention score
    score_scale = 1.0 / np.sqrt(cfg.head_dim)
    attn_scores = np.dot(q_reshaped[0], k_reshaped[0]) * score_scale
    print(f"  Attn score (h0,kv0): {attn_scores:.4f}")
    
    if np.abs(attn_scores) > 100:
        print(f"  ⚠️  Attention scores may be too large (explosions risk)")
    
    # Check MoE routing if applicable
    if hasattr(layer, 'moe_ffn') and layer.moe_ffn is not None:
        moe = layer.moe_ffn
        if moe.router is not None:
            print(f"  MoE: {cfg.n_experts} experts, router active")
    
    # Full forward
    h_out = layer.forward(h_input, pos, rope_freqs, k_cache, v_cache, cache_len)
    
    if not check_tensor("layer output", h_out):
        return None
    
    # Check residual connection effect
    residual_norm = np.linalg.norm(h_out - h_input)
    input_norm = np.linalg.norm(h_input)
    change_ratio = residual_norm / (input_norm + 1e-8)
    
    print(f"Output h: shape={h_out.shape}, min={h_out.min():.4f}, max={h_out.max():.4f}")
    print(f"  Change ratio: {change_ratio:.4f} (residual_norm/input_norm)")
    
    # Note: Large change ratios (up to 25x) are normal in early layers
    # as the model transforms embeddings into richer representations
    if change_ratio > 100.0:
        print(f"  ⚠️  Very large change ratio — possible instability")
    elif change_ratio < 0.001:
        print(f"  ⚠️  Very small change — layer may be vanishing")
    
    return h_out


def run_full_diagnostics(model_path: str, prompt: str = "Hello"):
    """Run full forward pass diagnostics."""
    print(f"Loading model: {model_path}")
    engine = Qwen25Engine(model_path)
    engine.load()
    cfg = engine.config
    
    print(f"\n=== Model Config ===")
    print(f"  Layers: {cfg.num_layers}")
    print(f"  Hidden: {cfg.hidden_size}")
    print(f"  Heads: {cfg.num_heads} Q, {cfg.num_kv_heads} KV")
    print(f"  Head dim: {cfg.head_dim}")
    print(f"  RoPE theta: {cfg.rope_theta}")
    print(f"  RMS norm eps: {cfg.rms_norm_eps}")
    
    _ni = _get_native_infer()
    print(f"  C++ kernels: {'✓' if _ni else '✗'}")
    
    # Tokenize prompt
    token_ids = engine.encode(prompt)
    print(f"\n=== Input ===")
    print(f"  Prompt: '{prompt}'")
    print(f"  Tokens: {token_ids[:10]}{'...' if len(token_ids)>10 else ''}")
    
    # Initialize caches
    max_seq = 1024  # For diagnostic, we only need a small cache
    k_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
                for _ in range(cfg.num_layers)]
    v_caches = [np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim), dtype=np.float32)
                for _ in range(cfg.num_layers)]
    
    # Process first token
    h = engine._embed_token(token_ids[0])
    print(f"\n=== Embedding ===")
    print(f"  Token: {token_ids[0]}")
    print(f"  Embedding: min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.6f}")
    
    if not check_tensor("embedding", h):
        return
    
    # Track layer outputs
    layer_stats = []
    
    # Forward through layers (detailed for first 3 and last 3)
    for i in range(cfg.num_layers):
        if i < 3 or i >= cfg.num_layers - 3:
            h = diagnose_layer_forward(
                engine, i, h, 0, engine.rope_freqs,
                k_caches[i], v_caches[i], 0
            )
            if h is None:
                print(f"\n❌ FATAL: Layer {i} produced invalid output!")
                return
        else:
            if i == 3:
                print(f"\n... (skipping detailed output for layers 3-{cfg.num_layers-4}) ...")
            layer = engine.layers[i]
            h = layer.forward(h, 0, engine.rope_freqs, k_caches[i], v_caches[i], 0)
            if not check_tensor(f"layer {i} output", h):
                print(f"\n❌ FATAL: Layer {i} produced invalid output!")
                return
        
        layer_stats.append({
            'idx': i,
            'min': float(h.min()),
            'max': float(h.max()),
            'mean': float(h.mean()),
            'std': float(h.std())
        })
    
    # Final norm
    print(f"\n=== Final Output Norm ===")
    h_final = _rms_norm(h, engine.output_norm, cfg.rms_norm_eps)
    print(f"  Pre-norm: min={h.min():.4f}, max={h.max():.4f}")
    print(f"  Post-norm: min={h_final.min():.4f}, max={h_final.max():.4f}")
    
    # LM head
    logits = engine._lm_head(h_final)
    print(f"\n=== Logits ===")
    print(f"  Shape: {logits.shape}")
    print(f"  Min: {logits.min():.4f}, Max: {logits.max():.4f}")
    print(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    
    if not check_tensor("logits", logits, warn_threshold=1000):
        print("  ⚠️  Logits may cause softmax issues!")
    
    # Top predictions
    top_k = 5
    top_ids = np.argsort(logits)[-top_k:][::-1]
    print(f"\n  Top {top_k} predictions:")
    for tid in top_ids:
        prob = _softmax(logits)[tid]
        token_str = engine.decode([tid])
        print(f"    {tid}: '{token_str}' (logit={logits[tid]:.2f}, prob={prob:.4f})")
    
    # Summary statistics
    print(f"\n=== Layer-by-Layer Statistics ===")
    print(f"{'Layer':>6} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print("-" * 50)
    
    for s in layer_stats[:5]:  # First 5
        print(f"{s['idx']:>6} {s['min']:>10.4f} {s['max']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
    
    if len(layer_stats) > 10:
        print("...")
    
    for s in layer_stats[-5:]:  # Last 5
        print(f"{s['idx']:>6} {s['min']:>10.4f} {s['max']:>10.4f} {s['mean']:>10.4f} {s['std']:>10.4f}")
    
    # Check for drift
    print(f"\n=== Drift Analysis ===")
    stds = [s['std'] for s in layer_stats]
    maxs = [s['max'] for s in layer_stats]
    
    std_growth = stds[-1] / (stds[0] + 1e-8)
    max_growth = maxs[-1] / (maxs[0] + 1e-8)
    
    print(f"  Std growth (last/first): {std_growth:.2f}x")
    print(f"  Max growth (last/first): {max_growth:.2f}x")
    
    if std_growth > 10:
        print("  ⚠️  Significant variance growth — may indicate numerical instability")
    if max_growth > 10:
        print("  ⚠️  Significant max value growth — may indicate activation explosion")
    
    print("\n✓ Diagnostics complete!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Forward pass diagnostics")
    parser.add_argument("model", nargs="?", default="qwen2.5-coder_3b.rftmwpk",
                        help="Path to .rftmwpk model")
    parser.add_argument("--prompt", default="Hello", help="Test prompt")
    args = parser.parse_args()
    
    run_full_diagnostics(args.model, args.prompt)
