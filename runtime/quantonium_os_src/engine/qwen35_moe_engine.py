"""Qwen3.5 MoE+SSM inference engine — mmap'd GGUF, our C++/AVX2 kernels.

SPDX-License-Identifier: LicenseRef-SPX-Proprietary
Copyright (C) 2025-2026 Luis M. Minier / quantoniumos

Architecture:
  - 40 layers, hidden_size=2048
  - Sliding layers (30): fused QKV → causal conv1d → Mamba SSM → gated output
  - Full attention layers (10, every 4th): separate Q(+gate)/K/V → GQA → output
  - ALL layers: MoE FFN (256 experts, top-8) + shared expert
  - Token embedding + output head via Q6_K dequant

Tensor access:
  Weights are read directly from the mmap'd GGUF file (zero-copy page faults).
  Dequantization + matmul use our C++/AVX2 fused kernels (q6k_matmul, q4k_matmul).
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# ── Project root for imports ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from quantonium_os_src.engine.gguf_mmap_engine import (
    GGUFMmapFile, GGUFTensor, GGUF_TYPE_Q4_K, GGUF_TYPE_Q6_K,
    GGUF_TYPE_F32, GGUF_TYPE_F16, GGUF_TYPE_BLOCK_SIZE, VALUES_PER_BLOCK,
)
from quantonium_os_src.engine.rftmw_numpy_engine import (
    Qwen35MoeConfig, QuantWeight,
    _rms_norm, _silu, _softmax, _apply_rope, _rope_freqs,
    _get_native_infer, _dequant_q6k_blocks, _dequant_q4k_blocks,
)

# RFT-RoPE integration: phi-basis complement for RoPE models
try:
    from algorithms.rft.variants.patent_variants import generate_rft_rope_complement
    _HAS_RFT_ROPE = True
except ImportError:
    _HAS_RFT_ROPE = False


# ═══════════════════════════════════════════════════════════════════
# Tensor loading helpers
# ═══════════════════════════════════════════════════════════════════

def _load_1d_from_gguf(gguf: GGUFMmapFile, name: str) -> Optional[np.ndarray]:
    """Load a 1-D tensor (norms, biases) from mmap'd GGUF → float32."""
    t = gguf.tensors.get(name)
    if t is None:
        return None
    raw = bytes(gguf.get_tensor_ptr(name))
    n_elem = t.ne[0]
    if t.ggml_type == GGUF_TYPE_F32:
        return np.frombuffer(raw, dtype=np.float32).copy()[:n_elem]
    elif t.ggml_type == GGUF_TYPE_F16:
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)[:n_elem]
    else:
        # Quantized 1-D: dequant
        bs = GGUF_TYPE_BLOCK_SIZE.get(t.ggml_type, 4)
        n_blocks = (n_elem + VALUES_PER_BLOCK - 1) // VALUES_PER_BLOCK
        raw_u8 = np.frombuffer(raw, dtype=np.uint8)
        blocks = raw_u8[:n_blocks * bs].reshape(n_blocks, bs)
        if t.ggml_type == GGUF_TYPE_Q6_K:
            return _dequant_q6k_blocks(blocks).ravel()[:n_elem].astype(np.float32)
        elif t.ggml_type == GGUF_TYPE_Q4_K:
            return _dequant_q4k_blocks(blocks).ravel()[:n_elem].astype(np.float32)
        return np.frombuffer(raw, dtype=np.float32).copy()[:n_elem]


def _load_2d_f32_from_gguf(gguf: GGUFMmapFile, name: str) -> Optional[np.ndarray]:
    """Load a 2-D F32/F16 tensor from mmap'd GGUF → ndarray (not QuantWeight)."""
    t = gguf.tensors.get(name)
    if t is None:
        return None
    raw = bytes(gguf.get_tensor_ptr(name))
    if t.ggml_type == GGUF_TYPE_F32:
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    elif t.ggml_type == GGUF_TYPE_F16:
        arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    else:
        arr = np.frombuffer(raw, dtype=np.float32).copy()
    if len(t.ne) >= 2:
        return arr.reshape(t.ne[1], t.ne[0])  # GGUF ne[0]=cols, ne[1]=rows
    return arr


def _load_2d_from_gguf(gguf: GGUFMmapFile, name: str) -> Optional[QuantWeight]:
    """Load a 2-D weight from mmap'd GGUF → QuantWeight (raw bytes, lazy dequant).

    For F32/F16 tensors, creates a QuantWeight with a pre-populated .dense field.
    """
    t = gguf.tensors.get(name)
    if t is None:
        return None
    raw = bytes(gguf.get_tensor_ptr(name))

    out_features = t.ne[1] if len(t.ne) > 1 else 1
    in_features = t.ne[0]

    # F32 / F16 tensors: load directly as dense matrix
    if t.ggml_type == GGUF_TYPE_F32:
        dense = np.frombuffer(raw, dtype=np.float32).copy().reshape(out_features, in_features)
        # Create a minimal QuantWeight shell with .dense set
        w = object.__new__(QuantWeight)
        w.out_features = out_features
        w.in_features = in_features
        w.bytes_per_row = in_features * 4
        w.raw = None
        w.quant_type = 'f32'
        w.block_size = 4
        w.dense = np.ascontiguousarray(dense, dtype=np.float32)
        return w
    elif t.ggml_type == GGUF_TYPE_F16:
        dense = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(out_features, in_features)
        w = object.__new__(QuantWeight)
        w.out_features = out_features
        w.in_features = in_features
        w.bytes_per_row = in_features * 2
        w.raw = None
        w.quant_type = 'f16'
        w.block_size = 2
        w.dense = np.ascontiguousarray(dense, dtype=np.float32)
        return w

    raw_u8 = np.frombuffer(raw, dtype=np.uint8).copy()
    bs = GGUF_TYPE_BLOCK_SIZE.get(t.ggml_type, 4)
    bytes_per_row = (in_features // VALUES_PER_BLOCK) * bs

    return QuantWeight(raw_u8, (out_features, bytes_per_row))


class MoeExpertWeights:
    """Manages 3-D expert weight tensors [in_features, ffn_size, num_experts].

    Stores the full raw bytes and extracts per-expert QuantWeight slices
    on demand (only dequanting selected experts during MoE forward).
    """
    __slots__ = ('raw', 'num_experts', 'ffn_rows', 'bytes_per_row',
                 'in_features', 'ggml_type')

    def __init__(self, gguf: GGUFMmapFile, name: str):
        t = gguf.tensors.get(name)
        if t is None:
            raise ValueError(f"Tensor {name} not found in GGUF")

        self.in_features = t.ne[0]
        self.ffn_rows = t.ne[1]
        self.num_experts = t.ne[2] if len(t.ne) > 2 else 1
        self.ggml_type = t.ggml_type

        bs = GGUF_TYPE_BLOCK_SIZE.get(t.ggml_type, 4)
        self.bytes_per_row = (self.in_features // VALUES_PER_BLOCK) * bs

        raw = bytes(gguf.get_tensor_ptr(name))
        self.raw = np.frombuffer(raw, dtype=np.uint8).copy()

    def expert_weight(self, expert_id: int) -> QuantWeight:
        """Extract a single expert's weight as QuantWeight for matmul."""
        start = expert_id * self.ffn_rows * self.bytes_per_row
        end = start + self.ffn_rows * self.bytes_per_row
        expert_raw = self.raw[start:end].copy()
        return QuantWeight(expert_raw, (self.ffn_rows, self.bytes_per_row))


# ═══════════════════════════════════════════════════════════════════
# SSM State
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SsmLayerState:
    """Per-layer SSM recurrent state."""
    h: np.ndarray             # (n_heads * head_dim,) = (4096,) for Qwen3.5
    conv_buf: np.ndarray      # (conv_kernel, qk_dim) = (4, 8192)
    conv_pos: int = 0


# ═══════════════════════════════════════════════════════════════════
# Per-layer weight container
# ═══════════════════════════════════════════════════════════════════

class Qwen35Layer:
    """Holds all weights for one transformer layer."""
    __slots__ = (
        'idx', 'is_full_attn',
        # Norms
        'attn_norm', 'post_attn_norm',
        # Full attention (every 4th layer)
        'attn_q', 'attn_k', 'attn_v', 'attn_output',
        'attn_q_norm', 'attn_k_norm',
        # Sliding SSM (other layers)
        'attn_qkv', 'attn_gate', 'ssm_conv1d',
        'ssm_a', 'ssm_alpha', 'ssm_beta', 'ssm_dt_bias',
        'ssm_norm', 'ssm_out',
        # MoE FFN (all layers)
        'ffn_gate_inp', 'ffn_gate_inp_shexp',
        'ffn_gate_exps', 'ffn_up_exps', 'ffn_down_exps',
        'ffn_gate_shexp', 'ffn_up_shexp', 'ffn_down_shexp',
    )

    def __init__(self, idx: int, is_full_attn: bool):
        self.idx = idx
        self.is_full_attn = is_full_attn
        # All set to None initially
        for attr in self.__slots__:
            if attr not in ('idx', 'is_full_attn'):
                setattr(self, attr, None)


# ═══════════════════════════════════════════════════════════════════
# Engine
# ═══════════════════════════════════════════════════════════════════

class Qwen35MoeEngine:
    """Full Qwen3.5 MoE+SSM engine using mmap'd GGUF + our C++/AVX2 kernels.

    Interface matches Qwen25Engine so LocalStreamWorker can use it directly.
    """

    def __init__(self, gguf_path: str, config: Optional[Qwen35MoeConfig] = None):
        self.gguf_path = gguf_path
        self.gguf = GGUFMmapFile()
        self.config = config or Qwen35MoeConfig()
        self.layers: List[Qwen35Layer] = []
        self.embed_weight: Optional[QuantWeight] = None
        self.output_weight: Optional[QuantWeight] = None
        self.output_norm: Optional[np.ndarray] = None
        self.rope_freqs: Optional[np.ndarray] = None
        self.rft_rope_basis: Optional[np.ndarray] = None  # RFT-RoPE complement basis
        self._tokenizer = None
        self._loaded = False

        # KV cache (only for full attention layers)
        self._k_cache: Dict[int, np.ndarray] = {}  # layer_idx → (max_seq, kv_heads, head_dim)
        self._v_cache: Dict[int, np.ndarray] = {}
        self._cache_len: int = 0

        # SSM state (only for sliding layers)
        self._ssm_states: Dict[int, SsmLayerState] = {}

        # Derived dims
        self._value_dim = 0  # value_dim = n_heads * head_v_dim
        self._key_dim = 0    # key_dim per K/Q in sliding layers
        self._conv_dim = 0   # conv1d dimension

    def load(self, on_stage: Optional[Callable] = None) -> None:
        """Mmap the GGUF and build layer weight refs."""
        if self._loaded:
            return

        if on_stage:
            on_stage("loading", {})

        t0 = time.perf_counter()

        if not self.gguf.open(self.gguf_path):
            raise RuntimeError(f"Failed to mmap GGUF: {self.gguf_path}")

        # Try to auto-detect config from GGUF metadata
        cfg_from_gguf = Qwen35MoeConfig.from_gguf(self.gguf_path)
        if cfg_from_gguf:
            self.config = cfg_from_gguf
            print(f"[qwen35] Config from GGUF: {self.config.num_layers}L, "
                  f"h={self.config.hidden_size}, heads={self.config.num_heads}/{self.config.num_kv_heads}, "
                  f"experts={self.config.num_experts}/{self.config.num_experts_per_tok}")

        cfg = self.config

        # Compute architecture dimensions from tensor shapes
        # attn_gate.weight [hidden, value_dim] for sliding layers
        gate_t = self.gguf.tensors.get("blk.0.attn_gate.weight")
        if gate_t:
            self._value_dim = gate_t.ne[1]  # e.g., 4096
        else:
            self._value_dim = cfg.ssm_inner_size  # fallback

        qkv_t = self.gguf.tensors.get("blk.0.attn_qkv.weight")
        if qkv_t:
            qkv_out = qkv_t.ne[1]  # 8192
            self._key_dim = (qkv_out - self._value_dim) // 2
        else:
            self._key_dim = cfg.num_kv_heads * cfg.head_dim

        self._conv_dim = 2 * self._value_dim  # = QKV output dim

        print(f"[qwen35] value_dim={self._value_dim}, key_dim={self._key_dim}, "
              f"conv_dim={self._conv_dim}")

        # Global tensors
        self.embed_weight = _load_2d_from_gguf(self.gguf, "token_embd.weight")
        self.output_weight = _load_2d_from_gguf(self.gguf, "output.weight")
        self.output_norm = _load_1d_from_gguf(self.gguf, "output_norm.weight")

        if on_stage:
            on_stage("loading", {"detail": "Loading layers..."})

        # Load each layer
        for i in range(cfg.num_layers):
            is_full = (i + 1) % cfg.full_attention_interval == 0
            layer = Qwen35Layer(i, is_full)

            prefix = f"blk.{i}."

            # Norms (all layers)
            layer.attn_norm = _load_1d_from_gguf(self.gguf, prefix + "attn_norm.weight")
            layer.post_attn_norm = _load_1d_from_gguf(self.gguf, prefix + "post_attention_norm.weight")

            if is_full:
                # Full attention layers
                layer.attn_q = _load_2d_from_gguf(self.gguf, prefix + "attn_q.weight")
                layer.attn_k = _load_2d_from_gguf(self.gguf, prefix + "attn_k.weight")
                layer.attn_v = _load_2d_from_gguf(self.gguf, prefix + "attn_v.weight")
                layer.attn_output = _load_2d_from_gguf(self.gguf, prefix + "attn_output.weight")
                layer.attn_q_norm = _load_1d_from_gguf(self.gguf, prefix + "attn_q_norm.weight")
                layer.attn_k_norm = _load_1d_from_gguf(self.gguf, prefix + "attn_k_norm.weight")

                # Init KV cache
                q_out = layer.attn_q.out_features if layer.attn_q else cfg.num_heads * cfg.head_dim
                # Infer num Q heads and V dim from tensor shapes
                n_q_heads = q_out // (2 * cfg.head_dim)  # Q is doubled (Q + gate)
                o_t = self.gguf.tensors.get(prefix + "attn_output.weight")
                v_total = o_t.ne[0] if o_t else cfg.num_heads * cfg.head_dim
                v_head_dim = v_total // n_q_heads if n_q_heads else cfg.head_dim

                max_seq = min(cfg.max_position, 8192)
                self._k_cache[i] = np.zeros((max_seq, cfg.num_kv_heads, cfg.head_dim),
                                            dtype=np.float32)
                self._v_cache[i] = np.zeros((max_seq, cfg.num_kv_heads, v_head_dim),
                                            dtype=np.float32)
            else:
                # Sliding SSM layers
                layer.attn_qkv = _load_2d_from_gguf(self.gguf, prefix + "attn_qkv.weight")
                layer.attn_gate = _load_2d_from_gguf(self.gguf, prefix + "attn_gate.weight")
                layer.ssm_conv1d = _load_2d_f32_from_gguf(self.gguf, prefix + "ssm_conv1d.weight")
                layer.ssm_a = _load_1d_from_gguf(self.gguf, prefix + "ssm_a")
                layer.ssm_alpha = _load_2d_from_gguf(self.gguf, prefix + "ssm_alpha.weight")
                layer.ssm_beta = _load_2d_from_gguf(self.gguf, prefix + "ssm_beta.weight")
                layer.ssm_dt_bias = _load_1d_from_gguf(self.gguf, prefix + "ssm_dt.bias")
                layer.ssm_norm = _load_1d_from_gguf(self.gguf, prefix + "ssm_norm.weight")
                layer.ssm_out = _load_2d_from_gguf(self.gguf, prefix + "ssm_out.weight")

                # Init SSM state: outer-product state [n_heads, head_dim, head_dim]
                # = [32, 128, 128] = 524288 floats per layer (stored flat)
                # conv_buf: [conv_kernel, qk_dim] = [4, 8192]
                n_h   = cfg.ssm_time_step_rank   # 32
                h_dim = cfg.ssm_state_size        # 128
                ssm_state_flat = n_h * h_dim * h_dim  # 32 * 128 * 128 = 524288
                qk_dim = layer.attn_qkv.out_features if layer.attn_qkv else 8192
                self._ssm_states[i] = SsmLayerState(
                    h=np.zeros(ssm_state_flat, dtype=np.float32),
                    conv_buf=np.zeros((cfg.ssm_conv_kernel, qk_dim), dtype=np.float32),
                    conv_pos=0,
                )

            # MoE FFN (all layers)
            layer.ffn_gate_inp = _load_2d_from_gguf(self.gguf, prefix + "ffn_gate_inp.weight")
            layer.ffn_gate_inp_shexp = _load_1d_from_gguf(self.gguf, prefix + "ffn_gate_inp_shexp.weight")

            # 3D expert weights — loaded as MoeExpertWeights for on-demand expert extraction
            try:
                layer.ffn_gate_exps = MoeExpertWeights(self.gguf, prefix + "ffn_gate_exps.weight")
                layer.ffn_up_exps = MoeExpertWeights(self.gguf, prefix + "ffn_up_exps.weight")
                layer.ffn_down_exps = MoeExpertWeights(self.gguf, prefix + "ffn_down_exps.weight")
            except ValueError as e:
                print(f"[qwen35] Warning: MoE weights missing for layer {i}: {e}")

            # Shared expert
            layer.ffn_gate_shexp = _load_2d_from_gguf(self.gguf, prefix + "ffn_gate_shexp.weight")
            layer.ffn_up_shexp = _load_2d_from_gguf(self.gguf, prefix + "ffn_up_shexp.weight")
            layer.ffn_down_shexp = _load_2d_from_gguf(self.gguf, prefix + "ffn_down_shexp.weight")

            self.layers.append(layer)

            if (i + 1) % 10 == 0:
                print(f"[qwen35] Loaded layer {i+1}/{cfg.num_layers}")

        # RoPE frequencies: use rope_dim_count (64), not head_dim (256)!
        # This affects frequency spacing: theta^(i/rope_dim) not theta^(i/head_dim)
        self.rope_freqs = _rope_freqs(cfg.rope_dim_count, min(cfg.max_position, 8192),
                                      cfg.rope_theta)

        # RFT-RoPE complement basis (patent variant for RoPE models)
        # Places phi-basis at golden-ratio offsets from RoPE frequencies
        if _HAS_RFT_ROPE:
            self.rft_rope_basis = generate_rft_rope_complement(cfg.head_dim, cfg.rope_theta)
            print(f"[qwen35] RFT-RoPE complement basis loaded: {self.rft_rope_basis.shape}")

        # Tokenizer
        self._load_tokenizer()

        elapsed = time.perf_counter() - t0
        n_tensors = len(self.gguf.tensors)
        print(f"[qwen35] Ready: {n_tensors} tensors mapped in {elapsed:.1f}s")
        print(f"[qwen35] Architecture: {cfg.num_layers}L, MoE({cfg.num_experts}/{cfg.num_experts_per_tok}), "
              f"SSM({cfg.ssm_state_size}), full_attn_every={cfg.full_attention_interval}")

        self._loaded = True
        if on_stage:
            on_stage("ready", {})

    # ──────────────────────────────────────────────────────────────
    # Tokenizer
    # ──────────────────────────────────────────────────────────────

    def _load_tokenizer(self):
        """Load tokenizer - build proper BPE from GGUF data for Qwen3.5."""
        import tokenizers
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        
        # Get GGUF tokens and merges
        gguf_tokens = self.gguf.metadata.get('tokenizer.ggml.tokens')
        gguf_merges = self.gguf.metadata.get('tokenizer.ggml.merges')
        
        if gguf_tokens and len(gguf_tokens) > 200000 and gguf_merges:
            # Decode tokens to strings
            self._gguf_tokens = [t.decode('utf-8', errors='replace') if isinstance(t, bytes) else t 
                                  for t in gguf_tokens]
            
            # Build vocab dict: {token_str: id}
            vocab = {tok: i for i, tok in enumerate(self._gguf_tokens)}
            
            # Decode merges: list of "token1 token2" strings
            merges = []
            for m in gguf_merges:
                if isinstance(m, bytes):
                    merges.append(m.decode('utf-8', errors='replace'))
                else:
                    merges.append(str(m))
            
            # Build BPE tokenizer
            merge_pairs = [(m.split()[0], m.split()[1]) for m in merges if len(m.split()) == 2]
            bpe = models.BPE(vocab=vocab, merges=merge_pairs)
            self._tokenizer = Tokenizer(bpe)
            
            # ByteLevel pre-tokenizer for normal tokens
            self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            
            # ByteLevel decoder to convert Ġ back to spaces
            self._tokenizer.decoder = decoders.ByteLevel()
            
            # Build special token mapping for manual handling in encode()
            # because ByteLevel corrupts <|...|> patterns
            self._special_tokens = {}  # {token_str: token_id}
            for i, tok in enumerate(self._gguf_tokens):
                if tok.startswith('<|') and tok.endswith('|>'):
                    self._special_tokens[tok] = i
            print(f"[qwen35] Registered {len(self._special_tokens)} special tokens (<|...|>)")
            
            print(f"[qwen35] Built BPE tokenizer: {len(self._gguf_tokens)} tokens, {len(merge_pairs)} merges")
            return
            
        # Fallback to HuggingFace tokenizer
        try:
            from tokenizers import Tokenizer
            # Try Qwen3.5 first
            try:
                self._tokenizer = Tokenizer.from_pretrained("Qwen/Qwen3-0.6B")
                print("[qwen35] Loaded Qwen3 tokenizer from HuggingFace")
                return
            except:
                pass
            # Try repo root
            tok_path = _PROJECT_ROOT / "tokenizer.json"
            if tok_path.exists():
                self._tokenizer = Tokenizer.from_file(str(tok_path))
                return
            self._tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B")
        except Exception as e:
            print(f"[qwen35] Warning: tokenizer not loaded: {e}")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs using BPE tokenizer.
        
        Handles special tokens (<|...|>) manually since ByteLevel corrupts them.
        """
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        
        # Handle special tokens manually
        import re
        special_pattern = re.compile(r'(<\|[^|]+\|>)')
        parts = special_pattern.split(text)
        
        result_ids = []
        for part in parts:
            if not part:
                continue
            if part in self._special_tokens:
                # Known special token - use direct ID
                result_ids.append(self._special_tokens[part])
            else:
                # Normal text - use BPE tokenizer
                encoded = self._tokenizer.encode(part)
                result_ids.extend(encoded.ids)
        
        return result_ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self._tokenizer is not None:
            # Use tokenizer's decoder which handles Ġ → space conversion
            return self._tokenizer.decode(ids)
        if hasattr(self, '_gguf_tokens') and self._gguf_tokens:
            # Fallback to raw lookup (won't convert Ġ properly)
            return ''.join(self._gguf_tokens[i] if 0 <= i < len(self._gguf_tokens) else '' for i in ids)
        raise RuntimeError("Tokenizer not loaded")

    # ──────────────────────────────────────────────────────────────
    # Forward pass: embedding
    # ──────────────────────────────────────────────────────────────

    def _embed(self, token_id: int) -> np.ndarray:
        """Look up token embedding → (hidden_size,)."""
        w = self.embed_weight
        if w.dense is not None:
            return w.dense[token_id].copy()

        # Extract one row from quantized weights
        row_raw = w.raw[token_id:token_id+1]  # (1, bytes_per_row)
        blocks = row_raw.reshape(-1, w.block_size)
        if w.quant_type == 'q6k':
            vals = _dequant_q6k_blocks(blocks)
        else:
            vals = _dequant_q4k_blocks(blocks)
        return vals.ravel()[:w.in_features].astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    # Forward pass: SSM layer (sliding) - Gated Delta Net
    # ──────────────────────────────────────────────────────────────

    def _forward_ssm_layer(self, x: np.ndarray, layer: Qwen35Layer) -> np.ndarray:
        """Forward through a sliding SSM layer — Mamba-2 diagonal state scan.

        Algorithm matches llama.cpp ggml_compute_forward_ssm_scan_f32:

          Projections from normed input:
            qk     = attn_qkv  @ normed  → [8192]   (Q=C + K=B, through conv+SiLU)
            X      = SiLU(attn_gate @ normed) → [4096]  (input values)

          Per-head scan (n_heads=32, dim_per_head=128, d_state=128):
            q[h], k[h] = split & L2-norm(conv_silu(qk))  # C, B: [32,128] each
            X[h]       = X reshaped                       # [32,128]
            dt[h]      = softplus(alpha[h] + dt_bias[h])
            decay[h]   = exp(dt[h] * A[h])                # A is negative → decay < 1
            x_dt       = X[h] * dt[h]                     # [128]
            
            # Diagonal state update (NOT outer-product):
            S[h]       = decay[h]*S[h] + outer(x_dt, B[h])  # [128,128]
            y[h]       = S[h] @ C[h]                         # [128]
            y[h]       = RMSNorm(y[h], ssm_norm)

          Output:
            ssm_output = ssm_out @ y_flat   # [4096→2048]
            return x + ssm_output
        """
        cfg = self.config
        h = x.copy()

        # Dimensions
        n_heads  = cfg.ssm_time_step_rank   # 32
        head_dim = cfg.ssm_state_size       # 128
        inner    = n_heads * head_dim       # 4096

        # Pre-attention norm
        normed = _rms_norm(h, layer.attn_norm, cfg.rms_norm_eps)

        # ── Projections ──
        qk = layer.attn_qkv.matmul(normed)   # [8192]  Q(C) + K(B)
        qk_dim = len(qk)

        # attn_gate → V values (NOT an output gate — see ssm_forward_qwen35 spec)
        v_flat = layer.attn_gate.matmul(normed)                         # [4096]
        # SiLU on V (Mamba-2 activates X before state update)
        v_flat = v_flat * (1.0 / (1.0 + np.exp(-np.clip(v_flat, -88, 88))))

        # ── Causal depthwise conv1d over qk ──
        state = self._ssm_states[layer.idx]
        conv_kernel = cfg.ssm_conv_kernel  # 4

        state.conv_buf[state.conv_pos] = qk
        state.conv_pos = (state.conv_pos + 1) % conv_kernel

        conv_out = np.zeros(qk_dim, dtype=np.float32)
        if layer.ssm_conv1d is not None:
            # ssm_conv1d: (qk_dim, conv_kernel) = (8192, 4)
            for k in range(conv_kernel):
                buf_idx = (state.conv_pos - 1 - k + conv_kernel) % conv_kernel
                conv_out += state.conv_buf[buf_idx] * layer.ssm_conv1d[:, k]
        # SiLU on conv output
        conv_out = conv_out * (1.0 / (1.0 + np.exp(-np.clip(conv_out, -88, 88))))

        # ── Split into Q (C) and K (B); L2-normalise per head ──
        q_heads = conv_out[:inner].reshape(n_heads, head_dim)    # [32, 128]
        k_heads = conv_out[inner:].reshape(n_heads, head_dim)    # [32, 128]
        q_norm = q_heads / (np.linalg.norm(q_heads, axis=1, keepdims=True) + 1e-6)
        k_norm = k_heads / (np.linalg.norm(k_heads, axis=1, keepdims=True) + 1e-6)

        # V reshaped (already SiLU'd)
        v_heads = v_flat.reshape(n_heads, head_dim)  # [32, 128]

        # ── SSM parameters ──
        alpha_raw = (layer.ssm_alpha.matmul(normed)
                     if layer.ssm_alpha is not None
                     else np.zeros(n_heads, dtype=np.float32))
        beta_raw  = (layer.ssm_beta.matmul(normed)
                     if layer.ssm_beta is not None
                     else np.zeros(n_heads, dtype=np.float32))
        dt_bias = (layer.ssm_dt_bias
                   if layer.ssm_dt_bias is not None
                   else np.zeros(n_heads, dtype=np.float32))
        A_log   = (layer.ssm_a
                   if layer.ssm_a is not None
                   else np.full(n_heads, -1.0, dtype=np.float32))

        # ── Mamba-2 SSM parameters (matches llama.cpp ggml_compute_forward_ssm_scan_f32) ──
        # dt = softplus(alpha + bias)
        dt = np.log1p(np.exp(np.clip(alpha_raw + dt_bias, -20, 20)))  # [32], ≥ 0
        
        # decay = exp(dt * A) where A is negative → decay ∈ (0, 1)
        # llama.cpp: dA = expf(dt_soft_plus * A[h])
        decay = np.exp(np.clip(dt * A_log, -20, 0))  # [32], ∈(0,1)
        
        # Note: beta_raw is NOT used in llama.cpp Mamba-2 scan. The B matrix
        # (k_norm) is used directly. beta may be for future extensions.

        # ── Mamba-2 SSM scan — diagonal state S[h] ∈ R^{dim × d_state} ──
        # State layout: [n_heads, head_dim, head_dim] = [32, 128, 128]
        # where dim=head_dim=128 (positions) and d_state=head_dim=128
        S = state.h.reshape(n_heads, head_dim, head_dim)  # [32, 128, 128]

        y_heads = np.empty((n_heads, head_dim), dtype=np.float32)
        for hi in range(n_heads):
            # x_dt = X * dt (llama.cpp: x_dt = x[ii] * dt_soft_plus)
            x_dt = v_heads[hi] * dt[hi]  # [128]
            
            # state[pos, state_dim] = state[pos, state_dim] * decay + B[state_dim] * x_dt[pos]
            # This is outer(x_dt, B): outer(x_dt, B)[pos, state_dim] = x_dt[pos] * B[state_dim]
            S[hi] = decay[hi] * S[hi] + np.outer(x_dt, k_norm[hi])  # [128, 128]
            
            # y[pos] = sum_state_dim(state[pos, state_dim] * C[state_dim]) = state[pos, :] @ C
            y_heads[hi] = S[hi] @ q_norm[hi]  # [128, 128] @ [128] = [128]

        state.h = S.ravel()

        # ── Apply ssm_norm as element-wise scale (NOT RMSNorm division) ──
        # llama.cpp ssm_scan doesn't do per-head RMSNorm inside the loop.
        # The ssm_norm may be a learned per-dimension scale factor.
        if layer.ssm_norm is not None:
            y_heads = y_heads * layer.ssm_norm   # [32,128] * [128] broadcast

        # ── Output projection ──
        ssm_output = (layer.ssm_out.matmul(y_heads.ravel())
                      if layer.ssm_out is not None
                      else np.zeros(cfg.hidden_size, dtype=np.float32))

        h = h + ssm_output
        return h

    # ──────────────────────────────────────────────────────────────
    # Forward pass: Full attention layer
    # ──────────────────────────────────────────────────────────────

    def _forward_attn_layer(self, x: np.ndarray, layer: Qwen35Layer,
                            pos: int) -> np.ndarray:
        """Forward through a full attention layer."""
        cfg = self.config
        h = x.copy()

        # Pre-attention norm
        normed = _rms_norm(h, layer.attn_norm, cfg.rms_norm_eps)

        # Q projection (doubled: Q + gate)
        q_gate_full = layer.attn_q.matmul(normed)   # [2 * num_heads * head_dim]
        q_dim = len(q_gate_full) // 2
        q_raw = q_gate_full[:q_dim]                   # [num_heads * head_dim]
        q_gate = q_gate_full[q_dim:]                  # [num_heads * head_dim] (gate)

        # K, V projections
        k_raw = layer.attn_k.matmul(normed)          # [kv_heads * head_dim]
        v_raw = layer.attn_v.matmul(normed)          # [kv_heads * v_head_dim]

        n_q_heads = q_dim // cfg.head_dim
        n_kv_heads = cfg.num_kv_heads
        head_dim = cfg.head_dim
        v_head_dim = len(v_raw) // n_kv_heads

        q = q_raw.reshape(n_q_heads, head_dim)
        k = k_raw.reshape(n_kv_heads, head_dim)
        v = v_raw.reshape(n_kv_heads, v_head_dim)

        # Per-head Q/K normalization
        if layer.attn_q_norm is not None:
            for hi in range(n_q_heads):
                rms = np.sqrt(np.mean(q[hi] ** 2) + 1e-6)
                q[hi] = (q[hi] / rms) * layer.attn_q_norm
        if layer.attn_k_norm is not None:
            for hi in range(n_kv_heads):
                rms = np.sqrt(np.mean(k[hi] ** 2) + 1e-6)
                k[hi] = (k[hi] / rms) * layer.attn_k_norm

        # Partial RoPE: rope_dim_count=64 of head_dim=256 are rotated.
        # Both C++ apply_rope and _apply_rope rotate all head_dim dims — wrong.
        # Apply in-place here using only the first rope_dim_count//2 freq pairs.
        _ni = _get_native_infer()
        if self.rope_freqs is not None and pos < len(self.rope_freqs):
            rope_dim = cfg.rope_dim_count   # 64
            d2 = rope_dim // 2              # 32 rotation pairs
            f = self.rope_freqs[pos, :d2]   # (32, 2) — first 32 of 128 pairs
            cos_f = f[:, 0]                 # (32,)
            sin_f = f[:, 1]                 # (32,)
            # Rotate dims 0..rope_dim-1 only; dims rope_dim..head_dim unchanged
            for heads in (q, k):
                x0 = heads[..., :d2].copy()
                x1 = heads[..., d2:rope_dim].copy()
                heads[..., :d2]        = x0 * cos_f - x1 * sin_f
                heads[..., d2:rope_dim] = x0 * sin_f + x1 * cos_f

        # KV cache
        li = layer.idx
        self._k_cache[li][self._cache_len] = k
        self._v_cache[li][self._cache_len] = v[:, :v_head_dim]
        seq_len = self._cache_len + 1

        # GQA attention
        heads_per_group = n_q_heads // n_kv_heads
        attn_out = np.zeros((n_q_heads, v_head_dim), dtype=np.float32)

        if _ni is not None and v_head_dim == head_dim:
            # Use C++ GQA when V dim matches head_dim
            try:
                attn_out = np.asarray(_ni.gqa_attention(
                    np.ascontiguousarray(q, dtype=np.float32),
                    np.ascontiguousarray(self._k_cache[li][:seq_len], dtype=np.float32),
                    np.ascontiguousarray(self._v_cache[li][:seq_len], dtype=np.float32),
                    n_q_heads, n_kv_heads, head_dim, seq_len))
            except Exception:
                attn_out = self._gqa_numpy(q, li, seq_len, n_q_heads, n_kv_heads,
                                           head_dim, v_head_dim, heads_per_group)
        else:
            attn_out = self._gqa_numpy(q, li, seq_len, n_q_heads, n_kv_heads,
                                       head_dim, v_head_dim, heads_per_group)

        # Apply Q gate (sigmoid gating of attention output)
        q_gate_sigmoid = 1.0 / (1.0 + np.exp(-np.clip(
            q_gate.reshape(n_q_heads, head_dim)[:, :v_head_dim], -88, 88)))
        attn_out = attn_out * q_gate_sigmoid

        # Output projection
        attn_flat = attn_out.reshape(-1)
        o = layer.attn_output.matmul(attn_flat)

        h = h + o
        return h

    def _gqa_numpy(self, q, li, seq_len, n_q_heads, n_kv_heads,
                    head_dim, v_head_dim, heads_per_group):
        """NumPy fallback for GQA."""
        attn_out = np.zeros((n_q_heads, v_head_dim), dtype=np.float32)
        for kv_idx in range(n_kv_heads):
            k_seq = self._k_cache[li][:seq_len, kv_idx]  # (seq, head_dim)
            v_seq = self._v_cache[li][:seq_len, kv_idx]  # (seq, v_head_dim)
            for h_off in range(heads_per_group):
                h_idx = kv_idx * heads_per_group + h_off
                if h_idx >= n_q_heads:
                    break
                scores = q[h_idx] @ k_seq.T / np.sqrt(np.float32(head_dim))
                weights = _softmax(scores)
                attn_out[h_idx] = weights @ v_seq
        return attn_out

    # ──────────────────────────────────────────────────────────────
    # Forward pass: MoE FFN
    # ──────────────────────────────────────────────────────────────

    def _forward_moe(self, x: np.ndarray, layer: Qwen35Layer) -> np.ndarray:
        """MoE FFN forward: route to top-k experts + shared expert."""
        cfg = self.config
        h = x.copy()

        # Post-attention norm
        normed = _rms_norm(h, layer.post_attn_norm, cfg.rms_norm_eps)

        # Router: select top-k experts
        router_logits = layer.ffn_gate_inp.matmul(normed)  # [num_experts]

        _ni = _get_native_infer()
        if _ni is not None:
            route = _ni.moe_topk_route(
                np.ascontiguousarray(router_logits, dtype=np.float32),
                cfg.num_experts, cfg.num_experts_per_tok)
            expert_ids = np.asarray(route["expert_ids"])
            expert_weights = np.asarray(route["expert_weights"])
            num_active = route["num_active"]
        else:
            # Python fallback for top-k routing
            top_k = cfg.num_experts_per_tok
            top_indices = np.argpartition(router_logits, -top_k)[-top_k:]
            top_logits = router_logits[top_indices]
            # Softmax over selected
            max_l = np.max(top_logits)
            exp_l = np.exp(top_logits - max_l)
            expert_weights = exp_l / np.sum(exp_l)
            expert_ids = top_indices
            num_active = top_k

        # Process selected experts
        moe_out = np.zeros(cfg.hidden_size, dtype=np.float32)
        for k in range(num_active):
            eid = int(expert_ids[k])
            weight = float(expert_weights[k])

            # Extract expert weights (on-demand dequant)
            gate_w = layer.ffn_gate_exps.expert_weight(eid)
            up_w = layer.ffn_up_exps.expert_weight(eid)
            down_w = layer.ffn_down_exps.expert_weight(eid)

            # SwiGLU: SiLU(x@gate) * (x@up)
            gate_val = gate_w.matmul(normed)
            up_val = up_w.matmul(normed)

            if _ni is not None:
                swiglu = _ni.silu_mul(
                    np.ascontiguousarray(gate_val, dtype=np.float32),
                    np.ascontiguousarray(up_val, dtype=np.float32))
                expert_out = down_w.matmul(np.asarray(swiglu))
            else:
                expert_out = down_w.matmul(_silu(gate_val) * up_val)

            moe_out += weight * expert_out

        # Shared expert contribution (gated by sigmoid)
        if layer.ffn_gate_inp_shexp is not None and layer.ffn_gate_shexp is not None:
            shared_gate_logit = np.dot(normed, layer.ffn_gate_inp_shexp)
            shared_gate = 1.0 / (1.0 + math.exp(-max(-88, min(88, shared_gate_logit))))

            if shared_gate > 0.01:
                sg = layer.ffn_gate_shexp.matmul(normed)
                su = layer.ffn_up_shexp.matmul(normed)
                if _ni is not None:
                    swiglu_sh = _ni.silu_mul(
                        np.ascontiguousarray(sg, dtype=np.float32),
                        np.ascontiguousarray(su, dtype=np.float32))
                    shared_out = layer.ffn_down_shexp.matmul(np.asarray(swiglu_sh))
                else:
                    shared_out = layer.ffn_down_shexp.matmul(_silu(sg) * su)
                moe_out += shared_gate * shared_out

        h = h + moe_out
        return h

    # ──────────────────────────────────────────────────────────────
    # Forward pass: full model
    # ──────────────────────────────────────────────────────────────

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        """Single-token forward pass → logits (vocab_size,)."""
        h = self._embed(token_id)

        for layer in self.layers:
            if layer.is_full_attn:
                h = self._forward_attn_layer(h, layer, pos)
            else:
                h = self._forward_ssm_layer(h, layer)

            h = self._forward_moe(h, layer)

        # Output norm + LM head
        h = _rms_norm(h, self.output_norm, self.config.rms_norm_eps)
        logits = self.output_weight.matmul(h)
        return logits

    def _sample(self, logits: np.ndarray, temperature: float = 0.7,
                top_k: int = 50) -> int:
        """Sample next token from logits."""
        if temperature < 0.01:
            return int(np.argmax(logits))

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0 and top_k < len(logits):
            indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -1e9)
            mask[indices] = logits[indices]
            logits = mask

        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)

        return int(np.random.choice(len(probs), p=probs))

    def reset_state(self):
        """Reset KV caches and SSM states for a new conversation."""
        self._cache_len = 0
        for li in self._k_cache:
            self._k_cache[li][:] = 0
            self._v_cache[li][:] = 0
        for li in self._ssm_states:
            s = self._ssm_states[li]
            s.h[:] = 0
            s.conv_buf[:] = 0
            s.conv_pos = 0

    # ──────────────────────────────────────────────────────────────
    # Generation API (matches Qwen25Engine interface)
    # ──────────────────────────────────────────────────────────────

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7,
                        on_stage: Optional[Callable] = None):
        """Generate tokens autoregressively, yielding decoded text.

        This is the interface LocalStreamWorker expects.
        """
        self.reset_state()

        if on_stage:
            on_stage("encoding", {})

        input_ids = self.encode(prompt)
        n_prompt = len(input_ids)

        if on_stage:
            on_stage("encoded", {"n_tokens": n_prompt})

        # Prefill: process all prompt tokens
        for i, tid in enumerate(input_ids):
            if i < n_prompt - 1:
                # Just run forward to build state, discard logits
                self.forward_token(tid, i)
                self._cache_len += 1
            else:
                # Last prompt token: get logits for first generation
                logits = self.forward_token(tid, i)
                self._cache_len += 1

        if on_stage:
            on_stage("decode_start", {})

        # Decode: generate tokens one by one
        eos_ids = {151643, 151644, 151645}  # Qwen EOS tokens
        for step in range(max_tokens):
            next_id = self._sample(logits, temperature)

            if next_id in eos_ids:
                break

            text = self.decode([next_id])
            yield text

            pos = n_prompt + step
            logits = self.forward_token(next_id, pos)
            self._cache_len += 1


# ═══════════════════════════════════════════════════════════════════
# Factory helper
# ═══════════════════════════════════════════════════════════════════

def create_qwen35_engine(gguf_path: str) -> Qwen35MoeEngine:
    """Create and load a Qwen35MoeEngine from a GGUF path."""
    engine = Qwen35MoeEngine(gguf_path)
    engine.load()
    return engine
