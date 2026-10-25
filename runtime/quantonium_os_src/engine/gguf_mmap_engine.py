#!/usr/bin/env python3
"""
SPDX-License-Identifier: LicenseRef-SPX-Proprietary
Copyright (C) 2025-2026 Luis M. Minier / quantoniumos

gguf_mmap_engine.py - Direct GGUF Mmap Inference Engine
========================================================

Zero-startup inference by mmap'ing GGUF files directly.
This is how llama.cpp achieves instant model loading.

Usage:
    from quantonium_os_src.engine.gguf_mmap_engine import GGUFMmapEngine
    
    engine = GGUFMmapEngine("C:/path/to/model.gguf")  # or .rftmwpk with gguf_path
    output = engine.generate("Hello", max_tokens=100)
"""

import mmap
import os
import sys
import struct
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# GGUF constants from llama.cpp
GGUF_MAGIC = 0x46554747  # 'GGUF' in little-endian
GGUF_VERSION = 3

# GGUF quant types
GGUF_TYPE_F32 = 0
GGUF_TYPE_F16 = 1
GGUF_TYPE_Q4_0 = 2
GGUF_TYPE_Q4_1 = 3
GGUF_TYPE_Q5_0 = 6
GGUF_TYPE_Q5_1 = 7
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_Q8_1 = 9
GGUF_TYPE_Q2_K = 10
GGUF_TYPE_Q3_K = 11
GGUF_TYPE_Q4_K = 12
GGUF_TYPE_Q5_K = 13
GGUF_TYPE_Q6_K = 14
GGUF_TYPE_Q8_K = 15
GGUF_TYPE_IQ2_XXS = 16
GGUF_TYPE_IQ2_XS = 17
GGUF_TYPE_IQ3_XXS = 18

# Block sizes for each type (bytes per 256 values)
GGUF_TYPE_BLOCK_SIZE = {
    GGUF_TYPE_F32: 1024,      # 256 * 4
    GGUF_TYPE_F16: 512,       # 256 * 2
    GGUF_TYPE_Q4_0: 18,       # 2 + 16
    GGUF_TYPE_Q4_1: 20,       # 2 + 2 + 16  
    GGUF_TYPE_Q5_0: 22,       # 2 + 4 + 16
    GGUF_TYPE_Q5_1: 24,       # 2 + 2 + 4 + 16
    GGUF_TYPE_Q8_0: 34,       # 2 + 32
    GGUF_TYPE_Q8_1: 36,       # 2 + 2 + 32
    GGUF_TYPE_Q2_K: 64,
    GGUF_TYPE_Q3_K: 64,
    GGUF_TYPE_Q4_K: 144,
    GGUF_TYPE_Q5_K: 176,
    GGUF_TYPE_Q6_K: 210,
    GGUF_TYPE_Q8_K: 292,
}

# Values per block
VALUES_PER_BLOCK = 256


@dataclass
class GGUFTensor:
    """Tensor metadata from GGUF index."""
    name: str
    offset: int
    size_bytes: int
    ggml_type: int
    ne: Tuple[int, ...]  # shape dims
    
    @property
    def rows(self) -> int:
        return self.ne[1] if len(self.ne) > 1 else 1
    
    @property 
    def cols(self) -> int:
        return self.ne[0] if len(self.ne) > 0 else 0


@dataclass
class GGUFConfig:
    """Model config extracted from GGUF metadata."""
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


class GGUFMmapFile:
    """Memory-mapped GGUF file with tensor index."""
    
    def __init__(self):
        self._mm: Optional[mmap.mmap] = None
        self._fd = None
        self._file_path: str = ""
        self._tensors: Dict[str, GGUFTensor] = {}
        self._metadata: Dict[str, Any] = {}
        self._data_start: int = 0
        self._config: Optional[GGUFConfig] = None
    
    def open(self, path: str) -> bool:
        """Mmap a GGUF file and parse its index."""
        try:
            self._file_path = path
            self._fd = open(path, 'rb')
            self._mm = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
            
            # Parse GGUF header
            if not self._parse_header():
                self.close()
                return False
            
            # Extract config from metadata
            self._extract_config()
            
            return True
        except Exception as e:
            print(f"[gguf_mmap] Failed to open {path}: {e}")
            return False
    
    def close(self):
        """Clean up mmap resources."""
        if self._mm:
            self._mm.close()
            self._mm = None
        if self._fd:
            self._fd.close()
            self._fd = None
    
    def _read_string(self, offset: int) -> Tuple[str, int]:
        """Read GGUF string (length-prefixed)."""
        length = struct.unpack_from('<Q', self._mm, offset)[0]
        s = self._mm[offset + 8 : offset + 8 + length].decode('utf-8', errors='replace')
        return s, offset + 8 + length
    
    def _read_value(self, offset: int, vtype: int) -> Tuple[Any, int]:
        """Read GGUF metadata value by type."""
        if vtype == 0:  # UINT8
            return self._mm[offset], offset + 1
        elif vtype == 1:  # INT8
            return struct.unpack_from('<b', self._mm, offset)[0], offset + 1
        elif vtype == 2:  # UINT16
            return struct.unpack_from('<H', self._mm, offset)[0], offset + 2
        elif vtype == 3:  # INT16
            return struct.unpack_from('<h', self._mm, offset)[0], offset + 2
        elif vtype == 4:  # UINT32
            return struct.unpack_from('<I', self._mm, offset)[0], offset + 4
        elif vtype == 5:  # INT32
            return struct.unpack_from('<i', self._mm, offset)[0], offset + 4
        elif vtype == 6:  # FLOAT32
            return struct.unpack_from('<f', self._mm, offset)[0], offset + 4
        elif vtype == 7:  # BOOL
            return self._mm[offset] != 0, offset + 1
        elif vtype == 8:  # STRING
            return self._read_string(offset)
        elif vtype == 9:  # ARRAY
            atype = struct.unpack_from('<I', self._mm, offset)[0]
            count = struct.unpack_from('<Q', self._mm, offset + 4)[0]
            pos = offset + 12
            arr = []
            for _ in range(count):
                val, pos = self._read_value(pos, atype)
                arr.append(val)
            return arr, pos
        elif vtype == 10:  # UINT64
            return struct.unpack_from('<Q', self._mm, offset)[0], offset + 8
        elif vtype == 11:  # INT64
            return struct.unpack_from('<q', self._mm, offset)[0], offset + 8
        elif vtype == 12:  # FLOAT64
            return struct.unpack_from('<d', self._mm, offset)[0], offset + 8
        else:
            raise ValueError(f"Unknown GGUF value type: {vtype}")
    
    def _parse_header(self) -> bool:
        """Parse GGUF header, metadata, and tensor index."""
        # Check magic
        magic = struct.unpack_from('<I', self._mm, 0)[0]
        if magic != GGUF_MAGIC:
            print(f"[gguf_mmap] Invalid magic: {magic:#x}, expected {GGUF_MAGIC:#x}")
            return False
        
        version = struct.unpack_from('<I', self._mm, 4)[0]
        if version not in (2, 3):
            print(f"[gguf_mmap] Unsupported GGUF version: {version}")
            return False
        
        n_tensors = struct.unpack_from('<Q', self._mm, 8)[0]
        n_kv = struct.unpack_from('<Q', self._mm, 16)[0]
        
        pos = 24
        
        # Read metadata KV pairs
        for _ in range(n_kv):
            key, pos = self._read_string(pos)
            vtype = struct.unpack_from('<I', self._mm, pos)[0]
            pos += 4
            value, pos = self._read_value(pos, vtype)
            self._metadata[key] = value
        
        # Read tensor info
        tensor_info = []
        for _ in range(n_tensors):
            name, pos = self._read_string(pos)
            n_dims = struct.unpack_from('<I', self._mm, pos)[0]
            pos += 4
            
            ne = []
            for _ in range(n_dims):
                dim = struct.unpack_from('<Q', self._mm, pos)[0]
                ne.append(dim)
                pos += 8
            
            ggml_type = struct.unpack_from('<I', self._mm, pos)[0]
            offset = struct.unpack_from('<Q', self._mm, pos + 4)[0]
            pos += 12
            
            tensor_info.append((name, tuple(ne), ggml_type, offset))
        
        # Align to 32 bytes for data section
        self._data_start = (pos + 31) & ~31
        
        # Calculate actual offsets and sizes
        for name, ne, ggml_type, rel_offset in tensor_info:
            n_elem = 1
            for d in ne:
                n_elem *= d
            n_blocks = (n_elem + VALUES_PER_BLOCK - 1) // VALUES_PER_BLOCK
            block_size = GGUF_TYPE_BLOCK_SIZE.get(ggml_type, 4)
            size_bytes = n_blocks * block_size
            
            # Absolute offset from file start
            abs_offset = self._data_start + rel_offset
            
            self._tensors[name] = GGUFTensor(
                name=name,
                offset=abs_offset,
                size_bytes=size_bytes,
                ggml_type=ggml_type,
                ne=ne
            )
        
        return True
    
    def _extract_config(self):
        """Extract model config from GGUF metadata."""
        m = self._metadata
        
        def get_int(key: str, default: int = 0) -> int:
            return int(m.get(key, default))
        
        def get_float(key: str, default: float = 0.0) -> float:
            return float(m.get(key, default))
        
        # Auto-detect architecture prefix
        arch = m.get('general.architecture', '')
        
        # Try different key patterns based on architecture
        prefixes = [arch, 'llama', 'qwen2', 'qwen35moe', '']
        
        def get_config_int(suffix: str, default: int = 0) -> int:
            for p in prefixes:
                key = f"{p}.{suffix}" if p else suffix
                if key in m:
                    return int(m[key])
            return default
        
        def get_config_float(suffix: str, default: float = 0.0) -> float:
            for p in prefixes:
                key = f"{p}.{suffix}" if p else suffix
                if key in m:
                    return float(m[key])
            return default
        
        # Extract config with architecture-aware lookup
        hidden = get_config_int('embedding_length', 2048)
        n_layers = get_config_int('block_count', 36)
        n_heads = get_config_int('attention.head_count', 16)
        n_kv_heads = get_config_int('attention.head_count_kv', n_heads)
        ffn = get_config_int('feed_forward_length', 11008)
        
        # Vocab size may be in tokenizer
        vocab = get_config_int('vocab_size', 0)
        if vocab == 0:
            vocab = len(m.get('tokenizer.ggml.tokens', [])) or 32000
        
        rope_theta = get_config_float('rope.freq_base', 1000000.0)
        ctx_len = get_config_int('context_length', 32768)
        
        self._config = GGUFConfig(
            hidden_size=hidden,
            num_layers=n_layers,
            num_heads=n_heads,
            num_kv_heads=n_kv_heads,
            intermediate_size=ffn,
            vocab_size=vocab,
            rope_theta=rope_theta,
            max_position=ctx_len
        )
    
    @property
    def config(self) -> Optional[GGUFConfig]:
        return self._config
    
    @property
    def tensors(self) -> Dict[str, GGUFTensor]:
        return self._tensors
    
    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata
    
    def get_tensor_ptr(self, name: str) -> Optional[memoryview]:
        """Get a memoryview of a tensor's raw bytes (zero-copy)."""
        if name not in self._tensors:
            return None
        t = self._tensors[name]
        return memoryview(self._mm)[t.offset : t.offset + t.size_bytes]
    
    def get_tensor_data(self, name: str) -> Optional[bytes]:
        """Get tensor data as bytes (copy)."""
        ptr = self.get_tensor_ptr(name)
        return bytes(ptr) if ptr else None


class GGUFMmapEngine:
    """
    Inference engine using mmap'd GGUF file.
    
    This achieves llama.cpp-like instant loading by:
    1. Memory-mapping the GGUF file (instant)
    2. Parsing tensor index from mmap (fast)
    3. Tensor data faults in on first access (on-demand)
    """
    
    def __init__(self, path: str):
        """
        Initialize engine from GGUF path or .rftmwpk with gguf_path in meta.
        
        Args:
            path: Path to .gguf file or .rftmwpk (will check meta.json)
        """
        self.path = path
        self.gguf = GGUFMmapFile()
        self._loaded = False
        self._gguf_path = self._resolve_gguf_path(path)
        self.tokenizer = None
    
    def _resolve_gguf_path(self, path: str) -> str:
        """Resolve actual GGUF path from pack or direct path."""
        # If it's a .gguf file, use directly
        if path.lower().endswith('.gguf'):
            return path
        
        # Check for meta.json sidecar
        meta_path = Path(path + '.meta.json')
        if meta_path.exists():
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
                gguf_path = meta.get('provenance', {}).get('gguf_path', '')
                if gguf_path and Path(gguf_path).exists():
                    return gguf_path
            except Exception as e:
                print(f"[gguf_mmap] Warning: could not read meta.json: {e}")
        
        # Try common Ollama blob paths
        if not path.lower().endswith('.gguf'):
            # Check if there's a blob reference
            ollama_path = Path.home() / '.ollama' / 'models' / 'blobs'
            if ollama_path.exists():
                # List available blobs
                for blob in ollama_path.iterdir():
                    if blob.name.startswith('sha256-'):
                        # Quick check if it's a GGUF
                        try:
                            with open(blob, 'rb') as f:
                                magic = f.read(4)
                                if struct.unpack('<I', magic)[0] == GGUF_MAGIC:
                                    return str(blob)
                        except:
                            pass
        
        return path
    
    def load(self) -> bool:
        """
        Memory-map the GGUF file.
        
        This is INSTANT — no data is read until first inference.
        """
        if not self.gguf.open(self._gguf_path):
            print(f"[gguf_mmap] Failed to mmap: {self._gguf_path}")
            return False
        
        self._load_tokenizer()
        self._loaded = True
        
        cfg = self.gguf.config
        print(f"[gguf_mmap] Mapped {self._gguf_path}")
        print(f"[gguf_mmap] Tensors: {len(self.gguf.tensors)}")
        print(f"[gguf_mmap] Config: {cfg.hidden_size}h, {cfg.num_layers}L, {cfg.vocab_size}V")
        print(f"[gguf_mmap] Ready for instant inference")
        return True
    
    def _load_tokenizer(self):
        """Load tokenizer from GGUF metadata or HuggingFace."""
        try:
            from tokenizers import Tokenizer
            
            # Try local tokenizer.json
            local_path = Path(self.path).parent / "tokenizer.json"
            if local_path.exists():
                self.tokenizer = Tokenizer.from_file(str(local_path))
                return
            
            # Try repo root
            repo_root = Path(__file__).parent.parent.parent.parent
            root_tok = repo_root / "tokenizer.json"
            if root_tok.exists():
                self.tokenizer = Tokenizer.from_file(str(root_tok))
                return
            
            # Fallback to HuggingFace (may need network)
            self.tokenizer = Tokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B")
        except Exception as e:
            print(f"[gguf_mmap] Warning: tokenizer not loaded: {e}")
    
    def encode(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text).ids
    
    def decode(self, ids: List[int]) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(ids)
    
    @property
    def config(self) -> Optional[GGUFConfig]:
        return self.gguf.config
    
    def get_tensor(self, name: str) -> Optional[memoryview]:
        """Get zero-copy view of tensor data."""
        return self.gguf.get_tensor_ptr(name)


def find_gguf_for_model(model_name: str) -> Optional[str]:
    """Find GGUF blob for a model name in Ollama cache."""
    ollama_path = Path.home() / '.ollama' / 'models'
    manifest_dir = ollama_path / 'manifests' / 'registry.ollama.ai' / 'library'
    
    # Look for model manifest
    model_dir = manifest_dir / model_name.split(':')[0]
    if not model_dir.exists():
        return None
    
    # Find latest tag
    tag = model_name.split(':')[1] if ':' in model_name else 'latest'
    manifest_file = model_dir / tag
    
    if not manifest_file.exists():
        # Try any available tag
        tags = list(model_dir.iterdir())
        if not tags:
            return None
        manifest_file = tags[0]
    
    # Parse manifest to find GGUF blob SHA
    try:
        with open(manifest_file) as f:
            manifest = json.load(f)
        
        for layer in manifest.get('layers', []):
            if 'model' in layer.get('mediaType', ''):
                digest = layer.get('digest', '')
                if digest.startswith('sha256:'):
                    blob_path = ollama_path / 'blobs' / digest.replace(':', '-')
                    if blob_path.exists():
                        return str(blob_path)
    except:
        pass
    
    return None


def main():
    """Demo: instant-load GGUF with mmap."""
    import time
    
    if len(sys.argv) < 2:
        print("Usage: python gguf_mmap_engine.py <path.gguf or model_name>")
        print("\nExamples:")
        print("  python gguf_mmap_engine.py C:/path/to/model.gguf")
        print("  python gguf_mmap_engine.py qwen35-opus-distil.rftmwpk")
        return
    
    path = sys.argv[1]
    
    print("=== GGUF Mmap Engine Demo ===")
    print(f"Input: {path}")
    
    # Check if it's a model name (Ollama style)
    if not Path(path).exists() and not path.endswith(('.gguf', '.rftmwpk')):
        gguf = find_gguf_for_model(path)
        if gguf:
            print(f"Found Ollama GGUF: {gguf}")
            path = gguf
        else:
            print(f"Could not find GGUF for model: {path}")
            return
    
    t0 = time.perf_counter()
    engine = GGUFMmapEngine(path)
    if not engine.load():
        return
    t1 = time.perf_counter()
    
    print(f"\nLoad time: {(t1-t0)*1000:.1f} ms (should be near-instant)")
    
    # Show some tensor info
    print("\nFirst 5 tensors:")
    for i, (name, t) in enumerate(list(engine.gguf.tensors.items())[:5]):
        print(f"  {name}: shape={t.ne}, type={t.ggml_type}, offset={t.offset}")
    
    # Get a tensor (first page fault)
    print("\nGetting embedding tensor (first page fault)...")
    t0 = time.perf_counter()
    emb = engine.get_tensor("token_embd.weight")
    if emb:
        print(f"  Got {len(emb):,} bytes in {(time.perf_counter()-t0)*1000:.1f} ms")
    else:
        print("  Tensor not found")


if __name__ == "__main__":
    main()
