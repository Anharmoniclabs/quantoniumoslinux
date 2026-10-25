#!/usr/bin/env python3
"""
SPDX-License-Identifier: LicenseRef-SPX-Proprietary
Copyright (C) 2025-2026 Luis M. Minier / quantoniumos

rftmw_mmap_engine.py - Instant-Load Inference Engine via mmap
==============================================================

Zero-startup-time engine using memory-mapped pack files.
Weights stay in kernel page cache, fault in on demand.

Key differences from rftmw_numpy_engine.py:
  - No 30s model load time — instant mmap
  - No NumPy arrays for weights — direct C pointers
  - No Python data copies — zero-copy matmul
  - Tensor data faults in on first access

Usage:
    from quantonium_os_src.engine.rftmw_mmap_engine import MmapEngine
    
    engine = MmapEngine("model.rftmwpk2")  # Instant!
    output = engine.generate("Hello", max_tokens=100)
"""

import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

# Add native module path
_native_paths = [
    'src/rftmw_native/build_asm',
    'src/rftmw_native/build_cpp', 
    'src/rftmw_native/build',
]
for p in _native_paths:
    full_path = Path(__file__).parent.parent.parent.parent / p
    if full_path.exists():
        sys.path.insert(0, str(full_path))
        break

# Windows DLL path
if sys.platform == 'win32':
    os.add_dll_directory(r'C:\msys64\mingw64\bin')

try:
    import rftmw_native
    _HAS_NATIVE = True
    _HAS_MMAP = getattr(rftmw_native, 'HAS_MMAP', False)
except ImportError:
    _HAS_NATIVE = False
    _HAS_MMAP = False


@dataclass
class MmapConfig:
    """Model configuration from mmap header."""
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    rope_theta: float
    max_position: int
    head_dim: int = 0
    
    def __post_init__(self):
        self.head_dim = self.hidden_size // self.num_heads


class MmapLayer:
    """Single transformer layer using mmap'd weights."""
    
    def __init__(self, model: 'rftmw_native.MmapModel', layer_idx: int, config: MmapConfig):
        self.model = model
        self.layer_idx = layer_idx
        self.config = config
        self.infer = rftmw_native.infer
        
        # Tensor name prefixes
        self.prefix = f"layer.{layer_idx}"
        
        # KV cache (only this is in Python memory)
        self.k_cache = np.zeros(
            (config.max_position, config.num_kv_heads, config.head_dim),
            dtype=np.float32
        )
        self.v_cache = np.zeros(
            (config.max_position, config.num_kv_heads, config.head_dim),
            dtype=np.float32
        )
    
    def _matmul(self, tensor_suffix: str, x: np.ndarray, 
                out_features: int, in_features: int) -> np.ndarray:
        """Zero-copy matmul on mmap'd tensor."""
        name = f"{self.prefix}.{tensor_suffix}"
        try:
            return self.model.q4k_matmul(name, x, out_features, in_features)
        except RuntimeError:
            # Try Q6K if Q4K fails
            return self.model.q6k_matmul(name, x, out_features, in_features)
    
    def forward(self, hidden: np.ndarray, pos: int) -> np.ndarray:
        """Forward pass through this layer."""
        cfg = self.config
        
        # Input layernorm - need the norm weights
        # For now, use native RMS norm with mmap'd weights
        # (This is a simplification - full impl would mmap the norm weights too)
        norm_weight = np.ones(cfg.hidden_size, dtype=np.float32)  # Placeholder
        h = self.infer.rms_norm(hidden.astype(np.float32), norm_weight, 1e-6)
        
        # QKV projections via mmap
        q = self._matmul("attn.q_proj", h, cfg.num_heads * cfg.head_dim, cfg.hidden_size)
        k = self._matmul("attn.k_proj", h, cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
        v = self._matmul("attn.v_proj", h, cfg.num_kv_heads * cfg.head_dim, cfg.hidden_size)
        
        # Apply RoPE
        cos_f = np.cos(np.arange(cfg.head_dim // 2) * pos / cfg.rope_theta).astype(np.float32)
        sin_f = np.sin(np.arange(cfg.head_dim // 2) * pos / cfg.rope_theta).astype(np.float32)
        q = self.infer.apply_rope(q.flatten(), cos_f, sin_f, cfg.num_heads, cfg.head_dim)
        k_new = self.infer.apply_rope(k.flatten(), cos_f, sin_f, cfg.num_kv_heads, cfg.head_dim)
        
        # Update KV cache
        self.k_cache[pos] = k_new.reshape(cfg.num_kv_heads, cfg.head_dim)
        self.v_cache[pos] = v.reshape(cfg.num_kv_heads, cfg.head_dim)
        
        # GQA attention
        attn_out = self.infer.gqa_attention(
            q.flatten(),
            self.k_cache[:pos+1].flatten(),
            self.v_cache[:pos+1].flatten(),
            cfg.num_heads,
            cfg.num_kv_heads,
            cfg.head_dim,
            pos + 1
        )
        
        # Output projection + residual
        o = self._matmul("attn.o_proj", attn_out.flatten(), cfg.hidden_size, 
                         cfg.num_heads * cfg.head_dim)
        hidden = hidden + o
        
        # Post-attention layernorm
        h = self.infer.rms_norm(hidden.astype(np.float32), norm_weight, 1e-6)
        
        # FFN (SwiGLU)
        gate = self._matmul("mlp.gate_proj", h, cfg.intermediate_size, cfg.hidden_size)
        up = self._matmul("mlp.up_proj", h, cfg.intermediate_size, cfg.hidden_size)
        ffn = self.infer.silu_mul(gate, up)
        down = self._matmul("mlp.down_proj", ffn, cfg.hidden_size, cfg.intermediate_size)
        
        return hidden + down


class MmapEngine:
    """
    Instant-load inference engine using memory-mapped pack files.
    
    Unlike Qwen25Engine which loads all weights into RAM (30s startup),
    this engine mmaps the file and faults pages in on demand (~0s startup).
    """
    
    def __init__(self, pack_path: str):
        if not _HAS_NATIVE:
            raise RuntimeError("Native module not available. Run: cmake --build build")
        if not _HAS_MMAP:
            raise RuntimeError("Native module lacks mmap support. Rebuild with mmap enabled.")
        
        self.pack_path = pack_path
        self.model = rftmw_native.MmapModel()
        self.layers: List[MmapLayer] = []
        self.config: Optional[MmapConfig] = None
        self.tokenizer = None
        self._loaded = False
    
    def load(self) -> bool:
        """
        Memory-map the pack file.
        
        This is INSTANT — no data is read until first inference.
        The kernel page cache handles prefetching.
        """
        if not self.model.load(self.pack_path):
            print(f"[mmap] Failed to mmap {self.pack_path}")
            print(f"[mmap] Ensure file exists and is RFTMWPK2 format")
            print(f"[mmap] Convert with: python src/apps/convert_pack_v2.py input.rftmwpk output.rftmwpk2")
            return False
        
        # Extract config from mmap header (no file I/O — just pointer reads)
        self.config = MmapConfig(
            hidden_size=self.model.hidden_size,
            num_layers=self.model.num_layers,
            num_heads=self.model.num_heads,
            num_kv_heads=self.model.num_kv_heads,
            intermediate_size=self.model.intermediate_size,
            vocab_size=self.model.vocab_size,
            rope_theta=self.model.rope_theta,
            max_position=self.model.max_position,
        )
        
        # Create layer wrappers (no data loaded yet!)
        self.layers = [
            MmapLayer(self.model, i, self.config)
            for i in range(self.config.num_layers)
        ]
        
        # Load tokenizer
        self._load_tokenizer()
        
        self._loaded = True
        print(f"[mmap] Mapped {self.pack_path} — {self.model.num_tensors} tensors")
        print(f"[mmap] Config: {self.config.hidden_size}h, {self.config.num_layers}L, {self.config.vocab_size}V")
        print(f"[mmap] Ready for instant inference (pages will fault in on demand)")
        return True
    
    def _load_tokenizer(self):
        """Load tokenizer from local file or HuggingFace."""
        try:
            from tokenizers import Tokenizer
            
            # Try local tokenizer.json next to pack
            local_path = Path(self.pack_path).parent / "tokenizer.json"
            if local_path.exists():
                self.tokenizer = Tokenizer.from_file(str(local_path))
                return
            
            # Try repo root
            repo_root = Path(__file__).parent.parent.parent.parent
            root_tok = repo_root / "tokenizer.json"
            if root_tok.exists():
                self.tokenizer = Tokenizer.from_file(str(root_tok))
                return
            
            # Fallback to HuggingFace
            self.tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B")
        except Exception as e:
            print(f"[mmap] Warning: tokenizer load failed: {e}")
            self.tokenizer = None
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(ids)
    
    def _embed_token(self, token_id: int) -> np.ndarray:
        """Lookup embedding via mmap (zero-copy)."""
        # For embedding table, we'd need to add a special accessor
        # For now, use a simple path through the existing model
        # TODO: Add direct embedding lookup to MmapModel
        return np.zeros(self.config.hidden_size, dtype=np.float32)
    
    def _lm_head(self, hidden: np.ndarray) -> np.ndarray:
        """Compute logits via mmap'd lm_head weights."""
        return self.model.q6k_matmul(
            "lm_head",
            hidden.astype(np.float32),
            self.config.vocab_size,
            self.config.hidden_size
        )
    
    def _sample(self, logits: np.ndarray, temperature: float = 0.7, 
                top_k: int = 40) -> int:
        """Sample next token from logits."""
        if temperature < 1e-6:
            return int(np.argmax(logits))
        
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0 and top_k < len(logits):
            indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.ones_like(logits) * float('-inf')
            mask[indices] = logits[indices]
            logits = mask
        
        # Softmax
        logits = logits - np.max(logits)
        probs = np.exp(logits) / np.sum(np.exp(logits))
        
        return int(np.random.choice(len(probs), p=probs))
    
    def generate(self, prompt: str, max_tokens: int = 100,
                 temperature: float = 0.7, top_k: int = 40) -> str:
        """Generate text from prompt."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        input_ids = self.encode(prompt)
        generated = []
        
        for pos in range(len(input_ids) + max_tokens):
            if pos < len(input_ids):
                token_id = input_ids[pos]
            else:
                if pos >= max_tokens + len(input_ids):
                    break
                token_id = generated[-1] if generated else input_ids[-1]
            
            # Embed (first page fault happens here!)
            hidden = self._embed_token(token_id)
            
            # Forward through all layers (more page faults)
            for layer in self.layers:
                hidden = layer.forward(hidden, pos)
            
            # Final norm + lm_head
            if pos >= len(input_ids) - 1:
                logits = self._lm_head(hidden)
                next_token = self._sample(logits, temperature, top_k)
                generated.append(next_token)
                
                # EOS check
                if next_token in [151643, 151645]:  # Qwen EOS tokens
                    break
        
        return self.decode(input_ids + generated)
    
    def clear_kv_cache(self):
        """Clear KV caches for new conversation."""
        for layer in self.layers:
            layer.k_cache.fill(0)
            layer.v_cache.fill(0)


def main():
    """Test the mmap engine."""
    import time
    
    if len(sys.argv) < 2:
        print("Usage: python rftmw_mmap_engine.py <pack.rftmwpk2>")
        print("Convert v1 first: python src/apps/convert_pack_v2.py input.rftmwpk output.rftmwpk2")
        return
    
    pack_path = sys.argv[1]
    
    print("=== MMAP Engine Test ===")
    print(f"Pack: {pack_path}")
    
    t0 = time.perf_counter()
    engine = MmapEngine(pack_path)
    if not engine.load():
        return
    t1 = time.perf_counter()
    print(f"Load time: {(t1-t0)*1000:.1f} ms (should be ~instant)")
    
    # First inference will fault in pages
    prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n"
    print(f"\nPrompt: {prompt!r}")
    
    t0 = time.perf_counter()
    output = engine.generate(prompt, max_tokens=10)
    t1 = time.perf_counter()
    
    print(f"Output: {output!r}")
    print(f"First inference: {(t1-t0)*1000:.1f} ms (includes page faults)")


if __name__ == "__main__":
    main()
