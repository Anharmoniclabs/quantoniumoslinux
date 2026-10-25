#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-quantoniumos-Claims-NC
# Copyright (C) 2025-2026 Luis M. Minier / quantoniumos
"""
RFTMW Serverless Inference Server
===================================

Ollama-compatible HTTP API backed by a .rftmwpk compressed weight pack.
Weights stay compressed in RAM (~887 MB for 3B params); only transiently
decompressed during each forward pass, then freed.

Endpoints:
    GET  /api/tags          — list available models
    POST /api/chat          — streaming chat completions
    GET  /health            — liveness probe

Usage:
    # Point at the pack file + HuggingFace model for tokenizer/config
    RFTMWPK_PATH=../../qwen2.5-coder_3b.rftmwpk \
    RFTMW_MODEL_ID=Qwen/Qwen2.5-Coder-3B \
    python rftmw_serve.py

    # Or with explicit port/host
    RFTMW_HOST=0.0.0.0 RFTMW_PORT=8000 python rftmw_serve.py

The mobile QU.AI app speaks the same Ollama protocol — just set the
server URL in Settings to point here.
"""
from __future__ import annotations

import json
import os
import struct
import sys
import time
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

# Resolve project root so imports work
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

# ── Configuration ──────────────────────────────────────────────────

PACK_PATH = os.environ.get(
    "RFTMWPK_PATH",
    str(_PROJECT_ROOT / "qwen2.5-coder_3b.rftmwpk"),
)
MODEL_ID = os.environ.get("RFTMW_MODEL_ID", "Qwen/Qwen2.5-Coder-3B")
HOST = os.environ.get("RFTMW_HOST", "0.0.0.0")
PORT = int(os.environ.get("RFTMW_PORT", "8000"))
MAX_TOKENS = int(os.environ.get("RFTMW_MAX_TOKENS", "256"))
PRELOAD_FP32 = os.environ.get("RFTMW_PRELOAD_FP32", "").lower() in ("1", "true", "yes")

# ── Engine singleton ───────────────────────────────────────────────

_engine = None
_engine_lock = Lock()
_pack_meta: Dict[str, Any] = {}


def _get_engine():
    """Lazy-load the inference engine with native C++ → NumPy fallback chain."""
    global _engine, _pack_meta

    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        # ── Load NumPy engine (C++ kernels lazy-load on first inference call) ──
        from quantonium_os_src.engine.rftmw_numpy_engine import Qwen25Engine

        print(f"[rftmw_serve] Loading pack: {PACK_PATH}")
        t0 = time.perf_counter()

        engine = Qwen25Engine(PACK_PATH)
        engine.load()

        elapsed = time.perf_counter() - t0
        n_layers = len(engine.layers)

        print(f"[rftmw_serve] Pack loaded: {n_layers} layers, {elapsed:.1f}s")
        if PRELOAD_FP32:
            print(f"[rftmw_serve] RFTMW_PRELOAD_FP32=1 — materializing weights (~12 GB RAM)")
            engine.preload_dense(free_raw=True)
            print(f"[rftmw_serve] FP32 fast path active (BLAS matmul, no per-token dequant)")
        else:
            print(f"[rftmw_serve] C++ inference kernels will activate on first chat request")
            print(f"[rftmw_serve] (set RFTMW_PRELOAD_FP32=1 for ~5-15x speedup on 64GB+ RAM)")

        # Read sidecar metadata if available
        sidecar = Path(PACK_PATH).with_suffix(".rftmwpk.meta.json")
        if not sidecar.exists():
            pack_name = Path(PACK_PATH).stem
            sidecar = Path(PACK_PATH).parent / f"quai-packs{pack_name}.rftmwpk.meta.json"
        if sidecar.exists():
            try:
                _pack_meta = json.loads(sidecar.read_text())
                print(f"[rftmw_serve] Sidecar loaded: {sidecar.name}")
            except Exception:
                pass

        _engine = engine
        return _engine


def _model_name_from_pack() -> str:
    """Extract a friendly model name from pack metadata or filename."""
    if _pack_meta:
        prov = _pack_meta.get("provenance", {})
        if prov.get("model_name"):
            return prov["model_name"]
    return Path(PACK_PATH).stem


def _build_prompt(messages: list[dict]) -> str:
    """Convert Ollama chat messages array to Qwen2.5 ChatML prompt."""
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
    parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)


# ── HTTP Handler ───────────────────────────────────────────────────

class RFTMWHandler(BaseHTTPRequestHandler):
    """Ollama-compatible HTTP handler for compressed inference."""

    def log_message(self, format, *args):
        # Quieter logging
        sys.stderr.write(f"[rftmw_serve] {format % args}\n")

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json_response(self, code: int, data: Any):
        body = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self._cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/tags":
            self._handle_tags()
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/api/chat":
            self._handle_chat()
        else:
            self._json_response(404, {"error": "not found"})

    def _handle_tags(self):
        """Return available models in Ollama /api/tags format."""
        name = _model_name_from_pack()
        self._json_response(200, {
            "models": [
                {
                    "name": name,
                    "model": name,
                    "modified_at": "2026-04-09T00:00:00Z",
                    "size": os.path.getsize(PACK_PATH) if os.path.exists(PACK_PATH) else 0,
                    "digest": "rftmwpk",
                    "details": {
                        "format": "rftmwpk",
                        "family": "qwen2.5",
                        "parameter_size": "3B",
                        "quantization_level": "INT8_ZLIB+RFT",
                    },
                }
            ]
        })

    def _handle_chat(self):
        """Handle POST /api/chat — streaming chat completions."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(content_length)
            req = json.loads(raw)
        except Exception:
            self._json_response(400, {"error": "invalid request body"})
            return

        messages = req.get("messages", [])
        if not messages:
            self._json_response(400, {"error": "no messages provided"})
            return

        stream = req.get("stream", False)
        options = req.get("options", {})
        max_tokens = int(options.get("max_new_tokens", options.get("num_predict", MAX_TOKENS)))
        temperature = float(options.get("temperature", 0.7))
        model_name = req.get("model", _model_name_from_pack())

        prompt = _build_prompt(messages)

        try:
            engine = _get_engine()
        except Exception as e:
            traceback.print_exc()
            self._json_response(500, {"error": f"Engine load failed: {e}"})
            return

        if stream:
            # True token-by-token streaming via generate_stream()
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self._cors_headers()
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            try:
                for token_text in engine.generate_stream(
                    prompt,
                    max_tokens=min(max_tokens, MAX_TOKENS),
                    temperature=max(temperature, 0.01),
                ):
                    chunk = json.dumps({
                        "model": model_name,
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "message": {"role": "assistant", "content": token_text},
                        "done": False,
                    }) + "\n"
                    self._write_chunk(chunk.encode("utf-8"))
            except Exception as e:
                traceback.print_exc()
                # Send error as a chunk so the client sees it
                err_chunk = json.dumps({
                    "model": model_name,
                    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "message": {"role": "assistant", "content": f"\n[error: {e}]"},
                    "done": False,
                }) + "\n"
                try:
                    self._write_chunk(err_chunk.encode("utf-8"))
                except Exception:
                    pass

            # Final done message
            done = json.dumps({
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": "stop",
            }) + "\n"
            self._write_chunk(done.encode("utf-8"))
            self._write_chunk(b"")  # terminal chunk
        else:
            try:
                response_text = engine.generate(
                    prompt,
                    max_tokens=min(max_tokens, MAX_TOKENS),
                    temperature=max(temperature, 0.01),
                )
            except Exception as e:
                traceback.print_exc()
                self._json_response(500, {"error": str(e)})
                return

            self._json_response(200, {
                "model": model_name,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "message": {"role": "assistant", "content": response_text},
                "done": True,
                "done_reason": "stop",
            })

    def _write_chunk(self, data: bytes):
        """Write a single HTTP chunked-transfer chunk."""
        self.wfile.write(f"{len(data):x}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()


# ── Threaded HTTP Server ───────────────────────────────────────────

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a separate thread to prevent deadlocks."""
    daemon_threads = True
    allow_reuse_address = True


# ── Main ───────────────────────────────────────────────────────────

def main():
    if not Path(PACK_PATH).exists():
        print(f"[rftmw_serve] ERROR: Pack file not found: {PACK_PATH}")
        print(f"[rftmw_serve] Set RFTMWPK_PATH to the .rftmwpk file location.")
        sys.exit(1)

    print(f"[rftmw_serve] RFTMW Serverless Inference")
    print(f"[rftmw_serve] Pack: {PACK_PATH}")
    print(f"[rftmw_serve] Model: {MODEL_ID}")
    print(f"[rftmw_serve] Endpoint: http://{HOST}:{PORT}")
    print(f"[rftmw_serve] Max tokens: {MAX_TOKENS}")
    print()

    # Pre-load engine synchronously (background preload was crashing on Windows)
    print("[rftmw_serve] Loading engine (this may take a minute)...")
    try:
        _get_engine()
        print("[rftmw_serve] Engine ready.")
    except Exception as e:
        print(f"[rftmw_serve] Engine load failed: {e}")
        import traceback; traceback.print_exc()
        print(f"[rftmw_serve] Will retry on first /api/chat request.")

    server = ThreadedHTTPServer((HOST, PORT), RFTMWHandler)
    print(f"[rftmw_serve] Listening on {HOST}:{PORT} (threaded)")
    print(f"[rftmw_serve] Ollama-compatible endpoints:")
    print(f"[rftmw_serve]   GET  /api/tags")
    print(f"[rftmw_serve]   POST /api/chat")
    print(f"[rftmw_serve]   GET  /health")
    print()

    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)  # let Ctrl+C work normally

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[rftmw_serve] Shutting down.")
    except Exception as e:
        print(f"[rftmw_serve] serve_forever() error: {e}")
        import traceback; traceback.print_exc()
    finally:
        server.server_close()
        print("[rftmw_serve] Server closed.")


if __name__ == "__main__":
    main()
