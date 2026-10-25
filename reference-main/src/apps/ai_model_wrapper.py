# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 Luis M. Minier / quantoniumos
"""Local LLM wrapper for on-device chat.

Design goals
- Works fully offline (no API calls).
- Uses HuggingFace Transformers locally.
- Supports optional LoRA adapters for local fine-tuning.

Notes
- Default model is kept small for CPU use. You can override with env vars.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np
from typing import Iterable, List, Optional, Tuple
import subprocess
from dataclasses import dataclass

# Add native module path
native_build_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rftmw_native', 'build'))
if native_build_path not in sys.path:
    sys.path.insert(0, native_build_path)

try:
    import rftmw_native
    HAS_NATIVE_RFT = True
    if os.getenv("QUANTONIUM_DEBUG_NATIVE") == "1":
        print("Native RFTMW engine loaded (AVX2/AVX-512)")
except ImportError:
    HAS_NATIVE_RFT = False
    # Only warn if not already printed
    pass

# Import fast quantum loader
try:
    from src.apps.fast_quantum_loader import fast_load_quantum_weights
    HAS_FAST_LOADER = True
except ImportError:
    HAS_FAST_LOADER = False

TORCH_AVAILABLE: Optional[bool] = None
torch = None  # type: ignore[assignment]
AutoModelForCausalLM = None  # type: ignore[assignment]
AutoTokenizer = None  # type: ignore[assignment]

_model = None
_tokenizer = None
_device = None


@dataclass(frozen=True)
class LocalModelCandidate:
    model_id: str
    source: str  # "repo_dir" | "hf_cache"
    detail: str


def _torch_import_healthcheck(timeout_s: float = 8.0) -> tuple[bool, str]:
    """Check torch import in a subprocess (torch DLL issues can terminate the process)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", "import torch; print(getattr(torch,'__version__','ok'))"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if proc.returncode == 0:
            return True, (proc.stdout or "").strip()
        msg = (proc.stderr or proc.stdout or "").strip()
        return False, msg or f"torch import failed (exit {proc.returncode})"
    except subprocess.TimeoutExpired:
        return False, "torch import timed out"
    except Exception as e:
        return False, str(e)


def _lazy_import_torch_transformers() -> None:
    global TORCH_AVAILABLE, torch, AutoModelForCausalLM, AutoTokenizer
    if TORCH_AVAILABLE is True:
        return
    if TORCH_AVAILABLE is False:
        raise RuntimeError("PyTorch/Transformers unavailable in this environment.")

    ok, detail = _torch_import_healthcheck()
    if not ok:
        TORCH_AVAILABLE = False
        raise RuntimeError(
            "PyTorch failed to import safely (subprocess check failed). "
            f"Details: {detail}"
        )

    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import AutoTokenizer as _AutoTokenizer

    torch = _torch
    AutoModelForCausalLM = _AutoModelForCausalLM
    AutoTokenizer = _AutoTokenizer
    TORCH_AVAILABLE = True


def clear_model_cache() -> None:
    global _model, _tokenizer, _device
    _model = None
    _tokenizer = None
    _device = None


def set_model_id(model_id: str) -> None:
    os.environ["QUANTONIUM_MODEL_ID"] = (model_id or "").strip()
    clear_model_cache()


def _discover_repo_hf_models() -> list[LocalModelCandidate]:
    """Discover HF-style models stored in-repo under ai/models/huggingface/*."""
    root = os.path.join("ai", "models", "huggingface")
    out: list[LocalModelCandidate] = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        # Heuristic: config.json present means transformers-compatible.
        if os.path.exists(os.path.join(p, "config.json")):
            out.append(LocalModelCandidate(model_id=p, source="repo_dir", detail=name))
    return out


def _discover_hf_cache_models() -> list[LocalModelCandidate]:
    """Discover models available in the local HF cache (HF_HOME/hub/models--*)."""
    hf_home = os.getenv("HF_HOME", "").strip() or os.path.join("ai", "hf_cache")
    hub = os.path.join(hf_home, "hub")
    out: list[LocalModelCandidate] = []
    if not os.path.isdir(hub):
        return out
    for name in sorted(os.listdir(hub)):
        if not name.startswith("models--"):
            continue
        # models--ORG--REPO -> ORG/REPO
        parts = name.split("--")
        if len(parts) < 3:
            continue
        org = parts[1]
        repo = "--".join(parts[2:])  # repo names can contain '--' in theory
        repo_id = f"{org}/{repo}"
        snapshots = os.path.join(hub, name, "snapshots")
        if os.path.isdir(snapshots) and any(os.path.isdir(os.path.join(snapshots, d)) for d in os.listdir(snapshots)):
            out.append(LocalModelCandidate(model_id=repo_id, source="hf_cache", detail=name))
    return out


def discover_local_models() -> list[LocalModelCandidate]:
    """Return a stable, de-duplicated list of local model candidates."""
    candidates = _discover_repo_hf_models() + _discover_hf_cache_models()
    seen: set[str] = set()
    out: list[LocalModelCandidate] = []
    for c in candidates:
        if c.model_id in seen:
            continue
        seen.add(c.model_id)
        out.append(c)
    return out


def get_configured_model_id() -> str:
    env = os.getenv("QUANTONIUM_MODEL_ID")
    if env and env.strip():
        val = env.strip()
        if val.lower() not in {"auto", "auto_best"}:
            return val

    # Auto-select best available local model.
    # Preference heuristic:
    # 1) Chat/instruct checkpoints in HF cache
    # 2) Any HF-cache checkpoint
    # 3) In-repo HF directory models
    candidates = discover_local_models()
    if candidates:
        def score(c: LocalModelCandidate) -> tuple[int, int, str]:
            mid = c.model_id.lower()
            is_chat = int(("chat" in mid) or ("instruct" in mid))
            is_cache = int(c.source == "hf_cache")
            return (-is_chat, -is_cache, mid)

        best = sorted(candidates, key=score)[0]
        return best.model_id

    # Fallback to a public id (may require network if not cached).
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def _resolve_quantum_weights_path() -> Optional[str]:
    env_path = os.getenv("QUANTONIUM_QUANTUM_WEIGHTS_JSON", "").strip()
    if env_path:
        return env_path

    candidates = [
        "ai/models/quantum/tinyllama_real_quantum_compressed.json",
        "data/parameters/quantum_models/tinyllama_real_quantum_compressed.json",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _pick_device() -> torch.device:
    _lazy_import_torch_transformers()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def apply_quantum_weights(model: torch.nn.Module, json_path: str):
    """Decompress and inject quantum weights from JSON into the model."""
    _lazy_import_torch_transformers()
    print(f"Decompressing quantum weights from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Group states by layer
        layer_states = {}
        for state in data.get('quantum_states', []):
            name = state['layer_name']
            if name not in layer_states:
                layer_states[name] = []
            layer_states[name].append(state)
            
        print(f"   Found quantum states for {len(layer_states)} layers.")
        
        # Apply to model
        print(f"   Reconstructing weights for {len(layer_states)} layers...")
        with torch.no_grad():
            for i, (name, param) in enumerate(model.named_parameters()):
                if name in layer_states:
                    states = layer_states[name]
                    if not states:
                        continue
                    if i % 5 == 0:
                        print(f"   ... processing layer {i+1}/{len(list(model.named_parameters()))}: {name} ({len(states)} states)")
                    
                    # Reconstruct using RFT Wave Summation: W = Sum(A * cos(F * t + P))
                    
                    # Create time index vector [0, 1, ..., numel-1]
                    numel = param.numel()
                    t = torch.arange(numel, device=param.device, dtype=torch.float32)
                    
                    # Initialize reconstruction
                    reconstructed = torch.zeros(numel, device=param.device, dtype=torch.float32)
                    
                    # Vectorized Batch Reconstruction (High Performance)
                    # Convert state lists to tensors for batch processing
                    s_amps = torch.tensor([s['amplitude'] for s in states], device=param.device, dtype=torch.float32)
                    s_freqs = torch.tensor([s['resonance_freq'] for s in states], device=param.device, dtype=torch.float32)
                    s_phases = torch.tensor([s['phase'] for s in states], device=param.device, dtype=torch.float32)
                    
                    used_native = False

                    # Use Native RFT Engine if available for O(N log N) speedup
                    if HAS_NATIVE_RFT:
                        # Native engine uses optimized assembly kernels (AVX-512/AVX2)
                        # This replaces the O(N*K) summation with O(N log N) transform
                        try:
                            # Create engine for this layer size
                            engine = rftmw_native.RFTMWEngine(max_size=numel)
                            
                            # Construct frequency domain spectrum
                            # Map resonance frequencies to spectral bins
                            # freq = 2*pi*k/N -> k = freq*N/(2*pi)
                            k_indices = (s_freqs * numel / (2 * 3.14159)).long() % numel
                            
                            # Add spectral components (amplitude * e^(i*phase))
                            # spectrum[k] += A * exp(i*P)
                            # Use complex64 for compatibility with native engine
                            spectral_vals = (s_amps * torch.exp(1j * s_phases)).to(torch.complex128)
                            
                            # Create spectrum on CPU for native interop
                            spectrum_np = torch.zeros(numel, dtype=torch.complex128).numpy()
                            
                            # Accumulate spectral values (using numpy for simplicity with native)
                            # Note: In production, we'd use index_add_ on GPU then transfer
                            k_np = k_indices.cpu().numpy()
                            vals_np = spectral_vals.cpu().numpy()
                            np.add.at(spectrum_np, k_np, vals_np)
                            
                            # Inverse RFT Transform (Fast C++ Execution)
                            reconstructed_np = engine.inverse(spectrum_np)
                            
                            # Copy back to tensor
                            reconstructed = torch.from_numpy(reconstructed_np).to(param.device, dtype=torch.float32)
                            used_native = True
                            
                        except Exception as e:
                            # print(f"Native RFT error: {e}, falling back to PyTorch batching")
                            # Fallback to batch processing below
                            pass

                    # Fallback if Native failed
                    if not used_native:
                        # Process in batches to manage memory while maximizing speed
                        batch_size = 256 # Increased batch size for speed
                        for i in range(0, len(states), batch_size):
                            b_amps = s_amps[i:i+batch_size].unsqueeze(1)   # [B, 1]
                            b_freqs = s_freqs[i:i+batch_size].unsqueeze(1) # [B, 1]
                            b_phases = s_phases[i:i+batch_size].unsqueeze(1) # [B, 1]
                            
                            # Compute wave batch: A * cos(F*t + P)
                            # Broadcasting: [B, 1] op [N] -> [B, N]
                            waves = b_amps * torch.cos(b_freqs * t + b_phases)
                            
                            # Sum across batch and add to reconstruction
                            reconstructed.add_(waves.sum(dim=0))
                    
                    # Reshape and inject into model
                    param.copy_(reconstructed.view(param.shape))
                    
        print("Quantum weights injected successfully (RFT wave reconstruction complete).")
        
    except Exception as e:
        print(f"Error applying quantum weights: {e}")
        print("   Falling back to standard weights.")


def get_model() -> Tuple[torch.nn.Module, "AutoTokenizer"]:
    """Load (and cache) the local model + tokenizer.

Env vars
- QUANTONIUM_MODEL_ID: HF model id (default: distilgpt2)
- QUANTONIUM_LORA_PATH: optional PEFT adapter path
- QUANTONIUM_LOCAL_ONLY=1: load only from local cache (no downloads)
"""
    global _model, _tokenizer, _device

    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    _lazy_import_torch_transformers()

    # Using TinyLlama for better chat quality (slower on CPU but smarter)
    model_id = get_configured_model_id()
    local_only = (os.getenv("QUANTONIUM_LOCAL_ONLY") == "1")
    lora_path = os.getenv("QUANTONIUM_LORA_PATH")

    if local_only:
        # Transformers may still attempt lightweight HEAD requests unless forced offline.
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    
    _device = _pick_device()

    print(f"Loading local model: {model_id} (device={_device})")
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_only)
        _model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_only)
    except Exception as e:
        if local_only:
            raise RuntimeError(
                "QUANTONIUM_LOCAL_ONLY=1 is set, but the model isn't available in the local HF cache. "
                "Run: python src/apps/cache_local_llm.py --model <id> --cache-dir ai/hf_cache (while online), "
                "then run again with HF_HOME=ai/hf_cache QUANTONIUM_LOCAL_ONLY=1. "
                f"Original error: {type(e).__name__}: {e}"
            ) from e
        raise

    # Ensure pad token exists for generation
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model.eval()
    _model.requires_grad_(False)
    _model.to(_device)

    quantum_path = _resolve_quantum_weights_path()
    if quantum_path:
        if not os.path.exists(quantum_path):
            raise RuntimeError(
                "QUANTONIUM_QUANTUM_WEIGHTS_JSON is set but file was not found: "
                f"{quantum_path}"
            )
        print(f"Loading quantum/RFT weight file: {quantum_path}")
        try:
            if HAS_FAST_LOADER:
                use_cache = os.getenv("QUANTONIUM_QUANTUM_CACHE", "1") != "0"
                loaded = fast_load_quantum_weights(quantum_path, _model, use_cache=use_cache)
                if not loaded:
                    apply_quantum_weights(_model, quantum_path)
            else:
                apply_quantum_weights(_model, quantum_path)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load quantum/RFT weights from {quantum_path}: {type(e).__name__}: {e}"
            ) from e
    else:
        print("Using standard model weights (fast mode)")

    # Optional: load LoRA adapter if configured
    if lora_path:
        try:
            from peft import PeftModel

            print(f"Loading LoRA adapter: {lora_path}")
            _model = PeftModel.from_pretrained(_model, lora_path)
            _model.eval()
            _model.to(_device)
        except Exception as e:
            print(f"LoRA adapter load failed: {type(e).__name__}: {e}")

    return _model, _tokenizer


def format_chat_prompt(
    user_text: str,
    history: Optional[Iterable[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
    max_turns: int = 6,
) -> str:
    """ChatML format for TinyLlama-Chat."""
    # TinyLlama uses ChatML format:
    # <|system|>\n{system}</s>\n<|user|>\n{user}</s>\n<|assistant|>\n
    
    parts: List[str] = []
    
    # System prompt
    if system_prompt:
        parts.append(f"<|system|>\n{system_prompt.strip()}</s>")
    else:
        parts.append("<|system|>\nYou are a helpful AI assistant running on quantoniumos.</s>")
    
    # History
    if history:
        turns = list(history)[-max_turns:]
        for u, a in turns:
            if u:
                parts.append(f"<|user|>\n{u.strip()}</s>")
            if a:
                parts.append(f"<|assistant|>\n{a.strip()}</s>")
    
    # Current user message
    parts.append(f"<|user|>\n{user_text.strip()}</s>")
    parts.append("<|assistant|>")
    
    return "\n".join(parts)


def format_plain_prompt(
    user_text: str,
    history: Optional[Iterable[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
    max_turns: int = 6,
) -> str:
    parts: List[str] = []
    if system_prompt:
        parts.append(f"System: {system_prompt.strip()}")

    if history:
        turns = list(history)[-max_turns:]
        for u, a in turns:
            if u:
                parts.append(f"User: {u.strip()}")
            if a:
                parts.append(f"Assistant: {a.strip()}")

    parts.append(f"User: {user_text.strip()}")
    parts.append("Assistant:")
    return "\n".join(parts)


def format_prompt_auto(
    user_text: str,
    history: Optional[Iterable[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
    max_turns: int = 6,
) -> str:
    model_id = get_configured_model_id().lower()
    if "tinyllama" in model_id and "chat" in model_id:
        return format_chat_prompt(user_text, history=history, system_prompt=system_prompt, max_turns=max_turns)
    return format_plain_prompt(user_text, history=history, system_prompt=system_prompt, max_turns=max_turns)


def generate_response(
    prompt: str,
    max_tokens: int = 160,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    model, tokenizer = get_model()

    inputs = tokenizer(prompt, return_tensors="pt").to(_device)
    input_length = inputs['input_ids'].shape[1]

    def _run_generate(*, do_sample: bool, temp: float, p: float):
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                min_new_tokens=1,
                do_sample=bool(do_sample),
                temperature=float(temp) if do_sample else None,
                top_p=float(p) if do_sample else None,
                repetition_penalty=1.10,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

    do_sample = os.getenv("QUANTONIUM_DO_SAMPLE", "1") != "0"
    do_sample = bool(do_sample and float(temperature) > 0.0)
    outputs = _run_generate(do_sample=do_sample, temp=float(temperature), p=float(top_p))

    # Only decode the NEW tokens (not the prompt)
    new_tokens = outputs[0][input_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    # If the model echoes ChatML tags, drop everything up to the first assistant content.
    # (Some models emit "<|assistant|>" or reprint the system/user blocks.)
    for prefix in ["<|assistant|>", "<|system|>", "<|user|>"]:
        if text.lstrip().startswith(prefix):
            # Remove leading tags/blocks by splitting on the last assistant tag if present.
            if "<|assistant|>" in text:
                text = text.split("<|assistant|>", 1)[-1]
            break

    # Trim at stop tokens
    for stop in ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "\nUser:", "\nSystem:"]:
        if stop in text:
            text = text.split(stop, 1)[0]

    text = text.strip()
    if not text:
        # Retry once with greedy decode; some chat checkpoints emit EOS immediately
        # under sampling parameters (especially on short factual prompts).
        outputs = _run_generate(do_sample=False, temp=0.0, p=1.0)
        new_tokens = outputs[0][input_length:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        for stop in ["</s>", "<|user|>", "<|system|>", "<|assistant|>", "\nUser:", "\nSystem:"]:
            if stop in text:
                text = text.split(stop, 1)[0].strip()

    return text
