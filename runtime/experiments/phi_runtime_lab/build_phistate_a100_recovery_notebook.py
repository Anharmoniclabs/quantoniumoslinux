"""Build the A100 recovery notebook with verified local storage and Drive API transport."""

import argparse
import ast
import base64
import hashlib
import json
import re
import textwrap
import zlib
from pathlib import Path


NAME = "PhiState_A100_Base_Recovery.ipynb"
PIN = "73edfa1a704e8d198269351ebe597e5ff8fed8ed"
BASE_MODULES = {
    "phistate_a100_base_journal.py", "phistate_base_windows.py",
    "phistate_base_reference.py", "phistate_base_cuda_train.py",
    "phistate_base_prerequisite.py", "device_preflight.py",
}
REQUIRED = BASE_MODULES | {
    "a100_run_guard.py", "phistate_checkpoint_io.py", "phistate_drive_transport.py",
    "phistate_a100_storage_recovery.py",
}

INTRO = """# PhiState A100 recovery and base training

Run the single code cell on your existing **A100** runtime. Approve the Google account containing your PhiState project when Colab asks. This notebook accesses Drive through its API and keeps the training files on Colab's local disk.

The original run's dense checkpoint was recovered and verified at **32 updates / 32,768 next-token targets**, including its model, optimizer and random-number-generator state. The worker resumes from this checkpoint unless a newer verified local recovery checkpoint exists. It verifies the archived 65K and 66K PhiState checkpoints and the repaired corpus. The six research modules, training schedules, seeds, optimizer settings and evaluation criteria are unchanged.

Checkpoints are written and verified locally every 32 updates. Verified Drive backups are made every 128 updates and at each arm's endpoint. If a backup fails, the worker stops with its local checkpoint retained under `/content/phistate_a100_recovery/`. Rerunning the cell in the same runtime resumes that local work. Local disk is temporary and is lost when Colab deletes the runtime, so wait for a verified Drive backup before intentionally releasing it.

The comparison runs dense and PhiState models on identical next-token windows across two seeds. Only a passing comparison proceeds to the bounded 66K base continuation. This remains the research-based base-training experiment; its report measures the actual result.

Leave `RELEASE_GPU_AFTER_FINISH` off if Wan or another job shares the runtime. It defaults to off and never releases the runtime after an error.
"""

LAUNCHER = r'''#@title Recover the checkpoint and continue on A100
RELEASE_GPU_AFTER_FINISH = False #@param {type:"boolean"}

import base64
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

PIN = "__PIN__"
REPO = "https://github.com/Anharmoniclabs/quantoniumoslinux.git"
LAB = Path("runtime/experiments/phi_runtime_lab")
WORK = Path("/content/phistate_a100_recovery")
REPORT = WORK / "project/phistate_a100_base_v1/report.json"
ENTRY = "phistate_a100_storage_recovery.py"

def require_a100():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select A100 GPU under Runtime > Change runtime type, then reconnect.")
    gpu = torch.cuda.get_device_properties(0)
    if "A100" not in gpu.name.upper():
        raise RuntimeError(f"Attached GPU is {gpu.name}. This journal requires A100.")
    with torch.inference_mode():
        value = torch.tensor([2., 3.], device="cuda:0").square().sum()
        torch.cuda.synchronize(0)
        if value.device.type != "cuda" or value.item() != 13.:
            raise RuntimeError("A100 CUDA arithmetic check failed.")
    print(f"{gpu.name} | {gpu.total_memory / 1024**3:.1f} GiB | PyTorch {torch.__version__} | CUDA {torch.version.cuda} | CUDA operation passed", flush=True)

def read_package():
    raw = zlib.decompress(base64.b64decode(SOURCES, validate=True))
    if hashlib.sha256(raw).hexdigest() != PACKAGE_SHA:
        raise RuntimeError("Embedded recovery package failed verification.")
    modules = json.loads(raw)
    if set(modules) != set(MODULE_HASHES):
        raise RuntimeError("Unexpected recovery package module set.")
    for name, source in modules.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.py", name):
            raise RuntimeError("Invalid recovery module filename.")
        if hashlib.sha256(source.encode()).hexdigest() != MODULE_HASHES[name]:
            raise RuntimeError(f"Recovery source mismatch: {name}")
        compile(source, name, "exec")
    return modules

def checkout(stage, modules):
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
        raise RuntimeError("Pinned PhiState checkout could not be verified.")
    overlays = {
        "device_preflight.py": "8ecd69302002cabc318f56cedb1c2b82647ee1343b58f69faaca3554ef24450d",
        "a100_run_guard.py": "d0b77375bc48f876474274e94a708215421b0c8649d2c914233654afd8ce23c1",
    }
    for name, source in modules.items():
        path = lab / name
        if name in overlays:
            if hashlib.sha256(path.read_bytes()).hexdigest() != overlays[name]:
                raise RuntimeError(f"Pinned dependency differs: {name}")
            mode = "wb"
        else:
            mode = "xb"
        with path.open(mode) as stream:
            stream.write(source.encode())
    return lab

def run_worker(lab):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               TOKENIZERS_PARALLELISM="false", PHISTATE_JOURNAL_COMMIT=PIN,
               PHISTATE_RECOVERY_PACKAGE_SHA=PACKAGE_SHA,
               PYTHONPATH=os.pathsep.join([str(lab), *filter(None, [os.environ.get("PYTHONPATH")])]))
    args = [sys.executable, "-u", str(lab / ENTRY), "--work", str(WORK)]
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
                        print("Stop requested. Waiting for the worker to save and exit.", flush=True)
                        try:
                            child.send_signal(signal.SIGINT)
                        except ProcessLookupError:
                            pass
    if code and not (interrupted and code in (-signal.SIGINT, 130)):
        raise RuntimeError(f"Recovery worker exited with status {code}. Local recovery files remain at {WORK}. See its last error above.")
    return interrupted

def launch():
    modules = read_package()
    if not PREREQUISITE_QUALIFIED:
        raise RuntimeError("The embedded research prerequisite is incomplete. No training started.")
    require_a100()
    packages = {"googleapiclient": "google-api-python-client", "google.auth": "google-auth",
                "numpy": "numpy", "sentencepiece": "sentencepiece"}
    missing = []
    for module, package in packages.items():
        try:
            present = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            present = False
        if not present:
            missing.append(package)
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
    from google.colab import auth
    print("Authorize the Google account holding your PhiState project for Drive API access.", flush=True)
    auth.authenticate_user()
    WORK.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="code_", dir=WORK))
    lab = checkout(stage, modules)
    (WORK / "code_directory.txt").write_text(str(lab) + "\n")
    print("Verified recovery package |", PACKAGE_SHA, "| local work", WORK, flush=True)
    if run_worker(lab):
        print("Worker stopped. Run this cell again in the same A100 runtime to resume.", flush=True)
        return
    if not REPORT.is_file():
        raise RuntimeError(f"Worker ended without its expected report: {REPORT}")
    status = json.loads(REPORT.read_text())
    print("Report:", REPORT, flush=True)
    for key in ("status", "outcome", "conclusion", "next_action"):
        if key in status:
            print(f"{key}: {status[key]}", flush=True)
    if RELEASE_GPU_AFTER_FINISH and status.get("status") == "complete":
        print("Worker completed its final backup. Releasing this Colab runtime.", flush=True)
        from google.colab import runtime
        runtime.unassign()

PACKAGE_SHA = "__PACKAGE_SHA__"
MODULE_HASHES = __MODULE_HASHES__
PREREQUISITE_QUALIFIED = __QUALIFIED__
SOURCES = (
__SOURCES__
)
launch()
'''


def build(source_dir, output):
    source_dir, output = Path(source_dir), Path(output)
    files = {}
    for name in sorted(REQUIRED):
        text = (source_dir / name).read_text(encoding="utf-8")
        compile(text, name, "exec")
        files[name] = text
    definitions = [node for node in ast.parse(files["phistate_base_prerequisite.py"]).body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "EVIDENCE" for t in node.targets)]
    if len(definitions) != 1 or not isinstance(definitions[0].value, ast.Call):
        raise ValueError("Expected one embedded evidence receipt.")
    receipt = json.loads(ast.literal_eval(definitions[0].value.args[0]))
    serialized = json.dumps(receipt["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    qualified = receipt["payload"]["qualified"]
    if (qualified is not True or hashlib.sha256(serialized).hexdigest() != receipt["sha256"]
            or receipt["payload"]["reference_module_sha256"] != hashlib.sha256(files["phistate_base_reference.py"].encode()).hexdigest()):
        raise ValueError("Research prerequisite or reference module failed verification.")
    raw = json.dumps(files, separators=(",", ":"), ensure_ascii=False).encode()
    encoded = base64.b64encode(zlib.compress(raw, level=9)).decode("ascii")
    code = (LAUNCHER.replace("__PIN__", PIN)
            .replace("__PACKAGE_SHA__", hashlib.sha256(raw).hexdigest())
            .replace("__MODULE_HASHES__", repr({n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()}))
            .replace("__QUALIFIED__", repr(qualified))
            .replace("__SOURCES__", "\n".join("    " + repr(p) for p in textwrap.wrap(encoded, 100))))
    compile(code, NAME, "exec")
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"colab": {"name": NAME, "provenance": [], "gpuType": "A100"},
                     "accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "name": "python3"},
                     "language_info": {"name": "python"}},
        "cells": [{"cell_type": "markdown", "id": "recovery-overview", "metadata": {}, "source": INTRO.splitlines(keepends=True)},
                  {"cell_type": "code", "id": "recovery-run", "metadata": {}, "execution_count": None,
                   "outputs": [], "source": code.splitlines(keepends=True)}],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return notebook


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / NAME)
    args = parser.parse_args()
    build(args.source_dir, args.output)
