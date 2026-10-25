# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 Luis M. Minier / quantoniumos
"""
Chatbox worker layer — in-process engine loading + streaming.

StreamWorker/HealthChecker remain for any HTTP-backed usage, but the
desktop PyQt app now loads the RFTMW engine in-process via
LocalEngineLoader and streams tokens through LocalStreamWorker.
"""

import json
import os
import subprocess
import time
import urllib.request
import urllib.error
import threading
from pathlib import Path
from typing import Any, List, Optional

from PyQt5.QtCore import QObject, pyqtSignal


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class StreamWorker(QObject):
    """Connects to rftmw_serve.py POST /api/chat and streams NDJSON tokens."""
    token_received = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, url: str, body: dict):
        super().__init__()
        self._url = url
        self._body = body
        self._cancelled = False

    def cancel(self):
        """Request graceful cancellation of the streaming loop."""
        self._cancelled = True

    def run(self):
        try:
            data = json.dumps(self._body).encode("utf-8")
            req = urllib.request.Request(
                self._url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                buf = b""
                while not self._cancelled:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("done"):
                            return
                        content = obj.get("message", {}).get("content", "")
                        if content:
                            self.token_received.emit(content)
        except Exception as e:
            if not self._cancelled:
                self.error.emit(str(e))
        finally:
            self.finished.emit()


class HealthChecker(QObject):
    """Polls rftmw_serve.py GET /api/tags every few seconds."""
    status_changed = pyqtSignal(bool, str)  # (connected, backend_type)

    def __init__(self, base_url: str):
        super().__init__()
        self._url = base_url
        self._connected = False
        self._backend = "demo"
        self._models: List[dict] = []
        self._pending_result = None  # (connected, backend) from bg thread
        self._checking = False

    @property
    def models(self):
        return self._models

    def check(self):
        """Run the HTTP probe on a background thread, then emit result on next call."""
        # If a previous background check finished, emit its result now (main thread).
        if self._pending_result is not None:
            connected, backend = self._pending_result
            self._pending_result = None
            if connected != self._connected or backend != self._backend:
                self._connected = connected
                self._backend = backend
                self.status_changed.emit(self._connected, self._backend)

        # Fire off the next probe if one isn't already running
        if not self._checking:
            self._checking = True
            threading.Thread(target=self._do_check, daemon=True).start()

    def _do_check(self):
        connected = False
        backend = "demo"
        try:
            req = urllib.request.Request(f"{self._url}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            models = data.get("models", [])
            self._models = models
            if models:
                fmt = models[0].get("details", {}).get("format", "")
                backend = "rftmw" if fmt == "rftmwpk" else "ollama"
            connected = True
        except Exception:
            self._models = []
        self._pending_result = (connected, backend)
        self._checking = False


# ── In-process engine workers (no HTTP) ────────────────────────────

class NativeServerEngine:
    """Persistent C++ Qwen35 runner used by the desktop chatbox."""

    def __init__(self, pack_path: str, max_seq: int = 1024):
        self.pack_path = str(pack_path)
        self.max_seq = max_seq
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stderr_handle = None

    @property
    def executable(self) -> Path:
        exe = PROJECT_ROOT / "src" / "rftmw_native" / "build_mmap" / "rftmw_qwen35_native.exe"
        if exe.exists():
            return exe
        return PROJECT_ROOT / "src" / "rftmw_native" / "build" / "rftmw_qwen35_native.exe"

    def load(self, timeout_s: float = 600.0):
        exe = self.executable
        if not exe.exists():
            raise FileNotFoundError(f"native executable not found: {exe}")

        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        err_path = log_dir / f"qshll_native_{time.strftime('%Y%m%d_%H%M%S')}_stderr.log"
        self._stderr_handle = open(err_path, "w", encoding="utf-8", errors="replace")
        cmd = [
            str(exe),
            "--pack", self.pack_path,
            "--serve",
            "--max-seq", str(self.max_seq),
        ]
        env = os.environ.copy()
        mingw_bin = r"C:\msys64\mingw64\bin"
        if Path(mingw_bin).exists() and mingw_bin.lower() not in env.get("PATH", "").lower():
            env["PATH"] = mingw_bin + os.pathsep + env.get("PATH", "")
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"native server exited with code {self.proc.returncode}; see {err_path}")
            assert self.proc.stdout is not None
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            if line.strip() == "READY":
                return
            if line.startswith("ERROR\t"):
                raise RuntimeError(bytes.fromhex(line.split("\t", 1)[1].strip()).decode("utf-8", "replace"))
        raise TimeoutError(f"native server did not become ready within {timeout_s:.0f}s; see {err_path}")

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7, on_stage=None):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("native server is not running")
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("native server pipes are unavailable")

        prompt_hex = prompt.encode("utf-8", "replace").hex()
        command = f"GENERATE\t{int(max_tokens)}\traw\t{prompt_hex}\t{temperature:.2f}\t0.9\n"
        with self._lock:
            self.proc.stdin.write(command)
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    if self.proc.poll() is not None:
                        raise RuntimeError(f"native server exited with code {self.proc.returncode}")
                    continue
                parts = line.rstrip("\n").split("\t")
                kind = parts[0] if parts else ""
                if kind == "STAGE" and len(parts) >= 3:
                    if on_stage:
                        stage = parts[1]
                        value = int(float(parts[2]))
                        info = {"n_tokens": value} if stage in {"encoded", "decode_start"} else {"value": value}
                        on_stage(stage, info)
                elif kind == "TOKEN" and len(parts) >= 5:
                    yield bytes.fromhex(parts[2]).decode("utf-8", "replace")
                elif kind == "DONE":
                    break
                elif kind == "ERROR" and len(parts) >= 2:
                    raise RuntimeError(bytes.fromhex(parts[1]).decode("utf-8", "replace"))

    def close(self):
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None


class SpectralNativeEngine:
    """RFTMW native Qwen25 engine with optional spectral FFN acceleration.

    Bypasses the Node.js server bridge entirely.  Spawns rftmw_qwen25_native
    with --gguf --spectral --serve and communicates via stdin/stdout
    tab-delimited protocol, streaming tokens back with zero HTTP overhead.
    """

    def __init__(self, gguf_path: str, max_seq: int = 2048,
                 spectral_keep: float = 0.80, mimosa_cache: int = 512,
                 kv_compress: bool = True, kv_keep: float = 0.60,
                 spectral: bool = False):
        self.gguf_path = str(gguf_path)
        self.max_seq = max_seq
        self.spectral_keep = spectral_keep
        self.mimosa_cache = mimosa_cache
        self.kv_compress = kv_compress
        self.kv_keep = kv_keep
        self.spectral = spectral
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._stderr_handle = None

    @property
    def executable(self) -> Path:
        # Prefer build_asm (has KV compress + layer streaming)
        exe = PROJECT_ROOT / "src" / "rftmw_native" / "build_asm" / "rftmw_qwen25_native.exe"
        if exe.exists():
            return exe
        exe = PROJECT_ROOT / "src" / "rftmw_native" / "build_qwen25" / "rftmw_qwen25_native.exe"
        if exe.exists():
            return exe
        return PROJECT_ROOT / "src" / "rftmw_native" / "build" / "rftmw_qwen25_native.exe"

    def load(self, timeout_s: float = 600.0):
        exe = self.executable
        if not exe.exists():
            raise FileNotFoundError(f"native executable not found: {exe}")
        log_dir = PROJECT_ROOT / "logs"
        log_dir.mkdir(exist_ok=True)
        err_path = log_dir / f"qshll_spectral_{time.strftime('%Y%m%d_%H%M%S')}_stderr.log"
        self._stderr_handle = open(err_path, "w", encoding="utf-8", errors="replace")
        cmd = [
            str(exe),
            "--gguf", self.gguf_path,
            "--serve",
            "--max-seq", str(self.max_seq),
            "--mimosa-cache", str(self.mimosa_cache),
        ]
        if self.spectral:
            cmd.extend(["--spectral", "--spectral-keep", str(self.spectral_keep)])
        if self.kv_compress:
            cmd.extend(["--kv-compress", "--kv-keep", str(self.kv_keep)])
        env = os.environ.copy()
        mingw_bin = r"C:\msys64\mingw64\bin"
        if Path(mingw_bin).exists() and mingw_bin.lower() not in env.get("PATH", "").lower():
            env["PATH"] = mingw_bin + os.pathsep + env.get("PATH", "")
        self.proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._stderr_handle,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"spectral engine exited with code {self.proc.returncode}; "
                    f"see {err_path}")
            assert self.proc.stdout is not None
            line = self.proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            if line.strip() == "READY":
                return
            if line.startswith("ERROR\t"):
                raise RuntimeError(
                    bytes.fromhex(line.split("\t", 1)[1].strip()).decode(
                        "utf-8", "replace"))
        raise TimeoutError(
            f"spectral engine not ready within {timeout_s:.0f}s; see {err_path}")

    def generate(self, prompt: str, max_tokens: int = 256,
                 temperature: float = 0.7) -> str:
        """Blocking generate — collect all tokens and return as string."""
        parts: list[str] = []
        for tok in self.generate_stream(prompt, max_tokens=max_tokens,
                                        temperature=temperature):
            parts.append(tok)
        return "".join(parts)

    def generate_stream(self, prompt: str, max_tokens: int = 256,
                        temperature: float = 0.7, on_stage=None):
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("spectral engine is not running")
        prompt_hex = prompt.encode("utf-8", "replace").hex()
        command = f"GENERATE\t{int(max_tokens)}\traw\t{prompt_hex}\t{temperature:.2f}\t0.9\n"
        with self._lock:
            assert self.proc.stdin is not None
            assert self.proc.stdout is not None
            self.proc.stdin.write(command)
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    if self.proc.poll() is not None:
                        raise RuntimeError(
                            f"spectral engine exited: {self.proc.returncode}")
                    continue
                parts = line.rstrip("\n").split("\t")
                kind = parts[0] if parts else ""
                if kind == "STAGE" and len(parts) >= 3 and on_stage:
                    on_stage(parts[1], {"value": int(float(parts[2]))})
                elif kind == "TOKEN" and len(parts) >= 5:
                    yield bytes.fromhex(parts[2]).decode("utf-8", "replace")
                elif kind == "DONE":
                    break
                elif kind == "ERROR" and len(parts) >= 2:
                    raise RuntimeError(
                        bytes.fromhex(parts[1]).decode("utf-8", "replace"))

    def close(self):
        proc = self.proc
        self.proc = None
        if proc is not None and proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.write("QUIT\n")
                    proc.stdin.flush()
                proc.wait(timeout=3)
            except Exception:
                proc.kill()
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None


class LocalEngineLoader(QObject):
    """Loads the RFTMW Qwen engine on a background thread — NATIVE KERNELS ONLY.

    Forces the C++/AVX2 fast path via RFTMW_REQUIRE_NATIVE=1.  Fused
    q4k_matmul / q6k_matmul kernels operate directly on quantized bytes
    (no FP32 preload needed), keeping startup fast and memory low.

    Emits:
        stage(str)              — "native", "loading", "preloading", "ready"
        ready(engine, name)     — engine ready for generate_stream()
        error(str)              — load or preload failure
    """
    stage = pyqtSignal(str)
    ready = pyqtSignal(object, str)
    error = pyqtSignal(str)

    def __init__(self, pack_path: str, spectral: bool = True,
                 kv_compress: bool = False, kv_keep: float = 0.60):
        super().__init__()
        self._pack_path = pack_path
        self._spectral = spectral
        self._kv_compress = kv_compress
        self._kv_keep = kv_keep

    @staticmethod
    def _resolve_pack_gguf(pack_path: str) -> Optional[Path]:
        """Resolve the source GGUF for a .rftmwpk pack via its sidecar meta.json.

        Checks both ``provenance.gguf_path`` and top-level ``gguf_path``,
        resolves relative paths, and falls back to Ollama SHA blob.
        Returns None if no GGUF can be found or the qwen25 exe doesn't exist.
        """
        exe = PROJECT_ROOT / "src" / "rftmw_native" / "build_asm" / "rftmw_qwen25_native.exe"
        if not exe.exists():
            exe = PROJECT_ROOT / "src" / "rftmw_native" / "build_qwen25" / "rftmw_qwen25_native.exe"
        if not exe.exists():
            return None
        pack = Path(pack_path)
        sidecar = pack.with_suffix(pack.suffix + ".meta.json")
        if not sidecar.exists():
            return None
        try:
            meta = json.loads(sidecar.read_text())
        except Exception:
            return None
        gp = (meta.get("provenance", {}).get("gguf_path", "")
              or meta.get("gguf_path", ""))
        sha = (meta.get("provenance", {}).get("gguf_file_sha256", "")
               or meta.get("gguf_file_sha256", ""))
        if gp:
            candidate = Path(gp)
            if candidate.exists():
                return candidate
            for base in (pack.parent, PROJECT_ROOT):
                resolved = base / candidate
                if resolved.exists():
                    return resolved
        if sha:
            blob = Path.home() / '.ollama' / 'models' / 'blobs' / f'sha256-{sha}'
            if blob.exists():
                return blob
        return None

    def run(self):
        pack_name = os.path.splitext(os.path.basename(self._pack_path))[0]
        # Hard-require native kernels — no silent NumPy fallback.
        os.environ["RFTMW_REQUIRE_NATIVE"] = "1"
        if os.getenv("QSHLL_NATIVE_SERVER", "1") != "0" and "qwen35" in pack_name.lower():
            try:
                self.stage.emit("native_server")
                max_seq = int(os.getenv("QSHLL_NATIVE_MAX_SEQ", "1024"))
                engine = NativeServerEngine(self._pack_path, max_seq=max_seq)
                engine.load()
                self.stage.emit("ready")
                self.ready.emit(engine, f"{pack_name}-native")
                return
            except Exception as e:
                self.error.emit(f"Native C++ server failed: {type(e).__name__}: {e}")
                return

        # ── Try SpectralNativeEngine for non-qwen35 packs that have a GGUF ──
        if os.getenv("QSHLL_NATIVE_SERVER", "1") != "0":
            gguf_resolved = self._resolve_pack_gguf(self._pack_path)
            if gguf_resolved is not None:
                try:
                    self.stage.emit("native_server")
                    max_seq = int(os.getenv("QSHLL_NATIVE_MAX_SEQ", "1024"))
                    engine = SpectralNativeEngine(
                        str(gguf_resolved), max_seq=max_seq,
                        spectral=self._spectral,
                        kv_compress=self._kv_compress,
                        kv_keep=self._kv_keep,
                    )
                    engine.load()
                    self.stage.emit("ready")
                    self.ready.emit(engine, f"{pack_name}-native")
                    return
                except Exception:
                    pass  # Fall through to Python engine

        try:
            self.stage.emit("native")
            from quantonium_os_src.engine.rftmw_numpy_engine import (
                Qwen25Engine, _get_native_infer, NativeKernelsUnavailable,
            )
            # Force native load up front so any failure happens here, not
            # buried in the first forward pass.
            _get_native_infer()
        except NativeKernelsUnavailable as e:
            self.error.emit(f"Native kernels unavailable:\n{e}")
            return
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")
            return

        try:
            self.stage.emit("loading")
            engine = Qwen25Engine(self._pack_path)
            engine.load()
        except Exception as e:
            self.error.emit(f"Pack load failed: {type(e).__name__}: {e}")
            return

        # Fused C++ kernels (q4k_matmul, q6k_matmul) work directly on
        # quantized bytes — no FP32 preload needed.  Skipping preload_dense()
        # saves ~60 s startup time and ~12-25 GB RAM.
        self.stage.emit("ready")
        self.ready.emit(engine, pack_name)


class LocalStreamWorker(QObject):
    """Streams tokens from an already-loaded engine's generate_stream().

    Signals:
        token_received(str)      — each decoded token text
        stage_changed(str, dict) — phase transition: "encoding", "encoded",
                                   "prefill_progress", "decode_start", "decoding"
        stats_updated(dict)      — running stats:
                                   {n_prompt, n_output, tokens_per_sec, elapsed_s}
        finished()               — stream complete (or cancelled)
        error(str)               — exception raised during generation
    """
    token_received = pyqtSignal(str)
    stage_changed = pyqtSignal(str, dict)
    stats_updated = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, engine: Any, prompt: str,
                 max_tokens: int = 256, temperature: float = 0.7):
        super().__init__()
        self._engine = engine
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import time
        t_start = time.perf_counter()
        t_decode_start: Optional[float] = None
        n_prompt = 0
        n_output = 0

        def on_stage(name: str, info: dict):
            nonlocal t_decode_start, n_prompt
            if name == "encoded":
                n_prompt = int(info.get("n_tokens", 0))
            elif name in ("decode_start", "decode"):
                t_decode_start = time.perf_counter()
            self.stage_changed.emit(name, dict(info))

        try:
            for token_text in self._engine.generate_stream(
                self._prompt,
                max_tokens=self._max_tokens,
                temperature=max(self._temperature, 0.01),
                on_stage=on_stage,
            ):
                if self._cancelled:
                    break
                if not token_text:
                    continue
                n_output += 1
                self.token_received.emit(token_text)

                # Emit running stats (tok/s over the decode phase only)
                now = time.perf_counter()
                decode_elapsed = (now - t_decode_start) if t_decode_start else 0.0
                tok_per_s = (n_output / decode_elapsed) if decode_elapsed > 0.01 else 0.0
                self.stats_updated.emit({
                    "n_prompt": n_prompt,
                    "n_output": n_output,
                    "tokens_per_sec": tok_per_s,
                    "elapsed_s": now - t_start,
                    "last_token": token_text,
                })
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()


class AgentStreamWorker(QObject):
    """Streams tokens from the agent tool-calling loop.

    Like LocalStreamWorker but wraps agent_stream() which can execute
    tools (read_file, list_dir, grep, python_eval, etc.) between
    generation rounds.

    Signals:
        token_received(str)      — each decoded token text
        stage_changed(str, dict) — inference phase transitions
        tool_event(str, str)     — tool execution: (event_type, detail)
                                   event_type: "start" | "result"
        reasoning_token(str)     — R1 reasoning/CoT tokens for thinking panel
        stats_updated(dict)      — running stats
        finished()               — stream complete
        error(str)               — exception raised
    """
    token_received = pyqtSignal(str)
    stage_changed = pyqtSignal(str, dict)
    tool_event = pyqtSignal(str, str)
    reasoning_token = pyqtSignal(str)
    stats_updated = pyqtSignal(dict)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, engine: Any, messages: list,
                 max_tokens: int = 512, temperature: float = 0.7):
        super().__init__()
        self._engine = engine
        self._messages = messages
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def run(self):
        import time
        from src.apps.chatbox_agent import agent_stream

        t_start = time.perf_counter()
        t_decode_start: Optional[float] = None
        n_output = 0

        def on_stage(name: str, info: dict):
            nonlocal t_decode_start
            if name in ("decode_start", "decode"):
                t_decode_start = time.perf_counter()
            self.stage_changed.emit(name, dict(info))

        def on_tool_start(name: str, args: dict):
            import json
            self.tool_event.emit("start", f"🔧 {name}({json.dumps(args, ensure_ascii=False)[:100]})")

        def on_tool_result(name: str, result: str):
            preview = result[:200].replace("\n", " ")
            self.tool_event.emit("result", f"✓ {name} → {preview}")

        def on_reasoning(token: str):
            self.reasoning_token.emit(token)

        try:
            for token_text in agent_stream(
                self._engine,
                self._messages,
                max_tokens=self._max_tokens,
                temperature=max(self._temperature, 0.01),
                on_stage=on_stage,
                on_tool_start=on_tool_start,
                on_tool_result=on_tool_result,
                on_reasoning=on_reasoning,
            ):
                if self._cancelled:
                    break
                if not token_text:
                    continue
                n_output += 1
                self.token_received.emit(token_text)

                now = time.perf_counter()
                decode_elapsed = (now - t_decode_start) if t_decode_start else 0.0
                tok_per_s = (n_output / decode_elapsed) if decode_elapsed > 0.01 else 0.0
                self.stats_updated.emit({
                    "n_prompt": 0,
                    "n_output": n_output,
                    "tokens_per_sec": tok_per_s,
                    "elapsed_s": now - t_start,
                    "last_token": token_text,
                })
        except Exception as e:
            if not self._cancelled:
                self.error.emit(f"{type(e).__name__}: {e}")
        finally:
            self.finished.emit()
