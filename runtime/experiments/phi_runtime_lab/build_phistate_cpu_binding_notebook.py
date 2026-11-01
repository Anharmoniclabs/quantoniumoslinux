"""Build the standalone one-cell CPU journal with the current binding experiment sources."""
import argparse
import base64
import hashlib
import json
import re
import textwrap
import zlib
from pathlib import Path

NAME = "PhiState_CPU_Binding_Journal.ipynb"
PIN = "73edfa1a704e8d198269351ebe597e5ff8fed8ed"
ENTRY = "phistate_binding_journal.py"
CALIBRATION = "phistate_copy_calibration.py"
REQUIRED = {ENTRY, CALIBRATION, "phistate_binding_data.py", "phistate_binding_fit.py",
            "phistate_binding_reference.py", "phistate_binding_bridge.py"}
MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
INTRO = """# PhiState CPU binding journal

Select **Runtime → Change runtime type → CPU**, then run the single code cell. When Colab prompts you, mount the Drive account containing `MyDrive/Quantonium/PhiState-L4`.

This journal tests one skill. Given two named values, can the model return the requested one when the question changes or the values trade places? A structured reference learns from explicitly parsed records, and a separate dense reference gets tokenized text. If the structured reference passes, the journal fits a separate copy of your frozen SFT-4,000 checkpoint using a fresh optimizer for up to **96 CPU updates**. It scores all four combinations of query and value assignment on training examples, fresh bindings, and fresh wording.

If the warm model misses its gate, the journal automatically tries single-value copying for up to **160 additional CPU updates**, with early stopping. If copying passes, a final **96-update two-value experiment** starts from the verified copy checkpoint and measures whether the model can now select the requested value. These checks measure a narrow retrieval skill. Allow **2 GiB of free Drive space** for the stage snapshots and the temporary checkpoint used during saves. The cell streams progress and saves its report to `whitespace_repair_v1/phistate_cpu_binding_v1/report.json`.

**After a disconnect**, reconnect to CPU and run the same cell. Only enable `RECOVER_DISCONNECTED` once the previous runtime has ended and a stale run lock is blocking recovery. CPU runtime varies with the assigned Colab machine."""
LAUNCHER = '''#@title Run PhiState binding test on CPU
RECOVER_DISCONNECTED = False #@param {type:"boolean"}

import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

PIN = "__PIN__"
SOURCE_HASH = "dcc79a5a3b05411725fefcefa04fd0aafd4393f4efb21e018c60da1e9a80d8ea"
REPO = "https://github.com/Anharmoniclabs/quantoniumoslinux.git"
LAB = Path("runtime/experiments/phi_runtime_lab")
MOUNT = Path("/content/drive")
ROOT = MOUNT / "MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1"
ENTRY = "phistate_binding_journal.py"
REPORT = ROOT / "phistate_cpu_binding_v1/report.json"

def require_cpu():
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        info = subprocess.run([nvidia, "-L"], capture_output=True, text=True, timeout=15)
        if info.returncode == 0 and re.search(r"(?m)^GPU \\d+:", info.stdout):
            raise RuntimeError("A GPU is allocated. Change runtime type to CPU, reconnect and run this cell.")
    if importlib.util.find_spec("torch") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                        "--index-url", "https://download.pytorch.org/whl/cpu", "torch"], check=True)
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("A CUDA device is attached. Change runtime type to CPU, reconnect and run this cell.")
    print("Verified CPU runtime | PyTorch", torch.__version__, flush=True)

def verify_checkpoint():
    checkpoint = ROOT / "coherence_execution_v1/source.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen SFT-4,000 checkpoint missing from the mounted Drive: {checkpoint}")
    with checkpoint.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    if checksum != SOURCE_HASH:
        raise RuntimeError(f"Unexpected frozen checkpoint SHA-256: {checksum}")
    print("Verified SFT-4,000 checkpoint |", checksum, flush=True)

def checkout(stage):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(stage), *args], text=True).strip()
    git("init", "-q")
    git("remote", "add", "origin", REPO)
    git("config", "remote.origin.promisor", "true")
    git("config", "remote.origin.partialclonefilter", "blob:none")
    git("sparse-checkout", "set", "--no-cone", "/runtime/experiments/phi_runtime_lab/", "/runtime/src/rftmw_native/")
    git("fetch", "--quiet", "--depth=1", "--filter=blob:none", "origin", PIN)
    git("checkout", "--quiet", "--detach", "FETCH_HEAD")
    lab = stage / LAB
    if git("rev-parse", "HEAD") != PIN or not (lab / "moe_train.py").is_file():
        raise RuntimeError("Public PhiState checkout could not be verified.")
    return lab

def unpack_scripts(lab):
    packed = zlib.decompress(base64.b64decode(SOURCES, validate=True))
    if hashlib.sha256(packed).hexdigest() != PACKAGE_SHA:
        raise RuntimeError("Embedded experiment package did not pass its integrity check.")
    modules = json.loads(packed)
    if ENTRY not in modules:
        raise RuntimeError("Embedded experiment entry point missing.")
    for name, source in modules.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\\.py", name):
            raise RuntimeError("Invalid embedded module filename.")
        compile(source, name, "exec")
        with (lab / name).open("xb") as stream:
            stream.write(source.encode("utf-8"))

def run_worker(lab):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               TOKENIZERS_PARALLELISM="false", PHISTATE_JOURNAL_COMMIT=PIN,
               PYTHONPATH=os.pathsep.join([str(lab), *filter(None, [os.environ.get("PYTHONPATH")])]))
    args = [sys.executable, "-u", str(lab / ENTRY), "--root", str(ROOT)]
    if RECOVER_DISCONNECTED:
        args.append("--recover-disconnected")
    interrupted = False
    with subprocess.Popen(args, cwd=lab, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, encoding="utf-8", errors="replace", bufsize=1, start_new_session=True) as child:
        with child.stdout:
            while True:
                try:
                    for line in child.stdout:
                        print(line, end="", flush=True)
                    code = child.wait()
                    break
                except KeyboardInterrupt:
                    if not interrupted:
                        interrupted = True
                        print("Stop requested. Waiting for the worker to exit.", flush=True)
                        try:
                            child.send_signal(signal.SIGINT)
                        except ProcessLookupError:
                            pass
    if code and not (interrupted and code in (-signal.SIGINT, 130)):
        raise RuntimeError(f"Binding worker exited with status {code}. See the last worker message above.")
    return interrupted

def launch():
    require_cpu()
    if not (MOUNT / "MyDrive").is_dir():
        from google.colab import drive
        drive.mount(str(MOUNT))
    verify_checkpoint()
    missing = [m for m in ("numpy", "sentencepiece") if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
    with tempfile.TemporaryDirectory(prefix="phistate_binding_", dir="/content") as tmp:
        lab = checkout(Path(tmp))
        unpack_scripts(lab)
        print("Starting | base", PIN, "| experiment", PACKAGE_SHA[:16], "| report", REPORT, flush=True)
        if run_worker(lab):
            print("Worker stopped. Rerun on CPU to resume.", flush=True)
            return
    if not REPORT.is_file():
        raise RuntimeError(f"Worker ended without its expected Drive report: {REPORT}")
    status = json.loads(REPORT.read_text(encoding="utf-8"))
    print(f"Drive report: {REPORT}", flush=True)
    for key in ("status", "outcome", "conclusion", "next_action"):
        if key in status:
            print(f"{key}: {status[key]}", flush=True)

# Compressed UTF-8 experiment sources; verified before running.
PACKAGE_SHA = "__SHA__"
SOURCES = (
__SOURCES__
)
launch()
'''


def build(source_dir, output, module_names=None):
    source_dir, output = Path(source_dir), Path(output)
    names = (sorted([p.name for p in source_dir.glob("phistate_binding_*.py")] + [CALIBRATION])
             if module_names is None else list(module_names))
    if not REQUIRED.issubset(names) or len(names) != len(set(names)) or not all(MODULE.fullmatch(n) for n in names):
        raise ValueError("Select distinct Python module basenames including all six binding experiment modules.")
    files = {}
    for name in sorted(names):
        source = (source_dir / name).read_bytes().decode("utf-8")
        compile(source, name, "exec")
        files[name] = source
    packed = json.dumps(files, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = base64.b64encode(zlib.compress(packed, level=9)).decode("ascii")
    code = (LAUNCHER.replace("__PIN__", PIN).replace("__SHA__", hashlib.sha256(packed).hexdigest())
            .replace("__SOURCES__", "\n".join("    " + repr(part) for part in textwrap.wrap(payload, 100))))
    compile(code, "PhiState CPU binding Colab launcher", "exec")
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata":{"colab": {"name": NAME, "provenance": []}, "kernelspec":{"display_name":"Python 3","name":"python3"},
                      "language_info":{"name":"python"}},
          "cells":[{"cell_type":"markdown","id":"binding-overview","metadata":{},"source":INTRO.splitlines(keepends=True)},
                   {"cell_type":"code","id":"binding-run","metadata":{},"execution_count":None,"outputs":[],
                    "source":code.splitlines(keepends=True)}]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(nb, indent=1, ensure_ascii=False)+"\n", encoding="utf-8")
    return nb


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / NAME)
    p.add_argument("--module", dest="modules", action="append", default=None)
    args = p.parse_args()
    build(args.source_dir, args.output, args.modules)
