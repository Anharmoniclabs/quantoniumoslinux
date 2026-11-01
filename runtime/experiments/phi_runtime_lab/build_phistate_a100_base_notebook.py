"""Build the single-cell A100 Colab training journal with its pinned experiment modules."""
import argparse
import ast
import base64
import hashlib
import json
import re
import textwrap
import zlib
from pathlib import Path

NAME = "PhiState_A100_Base_Comparison.ipynb"
PIN = "73edfa1a704e8d198269351ebe597e5ff8fed8ed"
ENTRY = "phistate_a100_base_journal.py"
REQUIRED = {ENTRY, "phistate_base_windows.py", "phistate_base_reference.py", "phistate_base_cuda_train.py",
            "phistate_base_prerequisite.py", "device_preflight.py"}
MODULE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\.py$")
INTRO = """# PhiState A100 base-model training comparison

Select **Runtime → Change runtime type → A100 GPU**, reconnect, and run the single code cell. When Colab asks, authorize the Drive account that contains `MyDrive/Quantonium/PhiState-L4`.

In prior CPU reference checks, a dense model improved its next-token predictions on natural code and text. A separate curriculum emphasizing the answer-changing token learned a narrow synthetic task. Those results are included in the notebook. The journal now measures ordinary next-token training on the A100. You can run it directly on your existing Drive project. The cell runs a CUDA operation on the attached A100 before mounting Drive. It launches PyTorch training with matched next-token prediction windows from your repaired **code and general-text** corpus, compares PhiState and a dense attention reference, and uses the measured results to decide whether the archived PhiState base checkpoint should receive a bounded continuation. It verifies the source weights and corpus in the worker. The source checkpoint and the completed chat experiment remain in their existing folders.

The worker verifies the archived 65K and 66K base checkpoints. It runs two independent seeds. For each seed the new PhiState and dense reference models receive the same training windows, 512 updates and 524,288 target tokens per model. A 512-update continuation from the archived 66K base is conditional on both architectures meeting the validation criteria on both seeds. The final report records the decision. [Pythia, matched data order](https://arxiv.org/abs/2304.01373) · [Chinchilla, compute and training-token budgets](https://arxiv.org/abs/2203.15556) · [Griffin, recurrence and local attention](https://arxiv.org/abs/2402.19427).

The worker streams progress and saves its checkpoints and `report.json` under `whitespace_repair_v1/phistate_a100_base_v1/`. To resume after a disconnect, reconnect to an A100 and run this cell again. Only enable `RECOVER_DISCONNECTED` after the previous runtime has ended and the worker reports a stale lock. Leave `RELEASE_GPU_AFTER_FINISH` off if Wan or another job shares this runtime. Enabling it releases the entire Colab runtime only after a complete report has been saved."""
LAUNCHER = '''#@title Train and compare base language models on A100
RELEASE_GPU_AFTER_FINISH = False #@param {type:"boolean"}
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
CORPUS_FINGERPRINT = "5a45ae09dfc42c6021202838439e77bd5cb05ff38d673d23e3219a1ecbc6c1d3"
REPO = "https://github.com/Anharmoniclabs/quantoniumoslinux.git"
LAB = Path("runtime/experiments/phi_runtime_lab")
MOUNT = Path("/content/drive")
ROOT = MOUNT / "MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1"
ENTRY = "phistate_a100_base_journal.py"
REPORT = ROOT / "phistate_a100_base_v1/report.json"

def require_a100():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU is attached. Choose A100 GPU under Runtime → Change runtime type, reconnect, and run this cell.")
    gpu = torch.cuda.get_device_properties(0)
    if "A100" not in gpu.name.upper():
        raise RuntimeError(f"Attached GPU is {gpu.name}. Select A100, reconnect, and rerun the cell.")
    with torch.inference_mode():
        value = torch.tensor([2., 3.], device="cuda:0").square().sum()
        torch.cuda.synchronize(0)
        if value.device.type != "cuda" or value.item() != 13.:
            raise RuntimeError("CUDA arithmetic check failed on the selected A100.")
    print(f"{gpu.name} | {gpu.total_memory / (1024**3):.1f} GiB | PyTorch {torch.__version__} | CUDA {torch.version.cuda} | CUDA operation passed", flush=True)

def drive_mounts():
    # Reading mount metadata avoids touching a broken FUSE mount.
    mounts = []
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        kind, source, *_ = right.split()
        if kind.startswith("fuse") and ("drive" in kind.lower() or "drive" in source.lower()):
            raw = left.split()[4]
            mounts.append(Path(re.sub(r"\\\\([0-7]{3})", lambda m: chr(int(m[1], 8)), raw)))
    return mounts

def ensure_drive():
    global MOUNT, ROOT, REPORT
    from google.colab import drive
    relative = Path("Quantonium/PhiState-L4/whitespace_repair_v1")

    def reuse():
        global MOUNT, ROOT, REPORT
        mounted = drive_mounts()
        for location in mounted:
            for home_name in ("MyDrive", "My Drive"):
                root = location / home_name / relative
                try:
                    with (root / "corpus_repaired_v1/manifest.json").open("rb") as stream:
                        if not stream.read(1):
                            continue
                except OSError:
                    continue
                MOUNT, ROOT = location, root
                REPORT = ROOT / "phistate_a100_base_v1/report.json"
                print("Drive project readable |", ROOT, flush=True)
                return True
        if mounted:
            raise RuntimeError(
                "Drive is mounted, but the PhiState corpus manifest is not readable. "
                "Check that the connected Google account owns MyDrive/Quantonium/PhiState-L4. "
                "The existing mount was left in place; training has not started.")
        return False

    if reuse():
        return
    # Colab's mount can stop other DriveFS processes. Never remount a live Drive.
    if MOUNT.is_symlink() or (MOUNT.exists() and
            (not MOUNT.is_dir() or next(MOUNT.iterdir(), None) is not None)):
        raise RuntimeError(f"Drive mount path contains local files: {MOUNT}. Nothing was removed; training has not started.")
    for attempt in (1, 2):
        print(f"Connecting Google Drive ({attempt}/2). Approve the Colab access prompt.", flush=True)
        try:
            drive.mount(str(MOUNT), force_remount=False, timeout_ms=120000)
        except (ValueError, OSError) as exc:
            if reuse():
                return
            if attempt == 2:
                raise RuntimeError(
                    "Google Drive connection failed twice; training has not started. "
                    "The A100 check passed. Open Colab's Files sidebar and use Mount Drive "
                    "with the account holding the project, then rerun this cell. "
                    "If mounting still fails, use Help > Send feedback with this error. "
                    f"Last Drive error: {exc}") from exc
            print("Drive startup failed. Retrying once in this A100 runtime.", flush=True)
            continue
        if reuse():
            return
        raise RuntimeError("Drive returned without a mounted project; training has not started.")

def verify_corpus():
    directory = ROOT / "corpus_repaired_v1"
    manifest = directory / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Repaired base corpus manifest missing from Drive: {manifest}")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    if record.get("format") != "PhiMoECorpus-v1" or record.get("data_fingerprint") != CORPUS_FINGERPRINT:
        raise RuntimeError(f"Unexpected repaired corpus identity at {manifest}")
    print("Found repaired base corpus |", CORPUS_FINGERPRINT, flush=True)

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

def read_package():
    packed = zlib.decompress(base64.b64decode(SOURCES, validate=True))
    if hashlib.sha256(packed).hexdigest() != PACKAGE_SHA:
        raise RuntimeError("Embedded experiment package did not pass its integrity check.")
    modules = json.loads(packed)
    expected = hashlib.sha256(modules["phistate_base_prerequisite.py"].encode()).hexdigest()
    if expected != PREREQUISITE_SOURCE_HASH:
        raise RuntimeError("Embedded prerequisite decision differs from the reference receipt.")
    if ENTRY not in modules:
        raise RuntimeError("Embedded experiment entry point missing.")
    return modules

def unpack_scripts(lab, modules):
    for name, source in modules.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\\.py", name):
            raise RuntimeError("Invalid embedded module filename.")
        compile(source, name, "exec")
        path = lab / name
        if name == "device_preflight.py":
            pinned_hash = "8ecd69302002cabc318f56cedb1c2b82647ee1343b58f69faaca3554ef24450d"
            if hashlib.sha256(path.read_bytes()).hexdigest() != pinned_hash:
                raise RuntimeError("Pinned base preflight dependency has changed.")
            mode = "wb"
        else:
            mode = "xb"
        with path.open(mode) as stream:
            stream.write(source.encode("utf-8"))

def run_worker(lab):
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
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
        raise RuntimeError(f"Base training worker exited with status {code}. See the last worker message above.")
    return interrupted

def launch():
    modules = read_package()
    if not PREREQUISITE_QUALIFIED:
        raise RuntimeError("Reference qualification is incomplete for this snapshot. Leave the A100 disconnected; no Drive mount, CUDA probe, or training has been requested.")
    require_a100()
    ensure_drive()
    verify_corpus()
    missing = [m for m in ("numpy", "sentencepiece") if importlib.util.find_spec(m) is None]
    if missing:
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *missing], check=True)
    with tempfile.TemporaryDirectory(prefix="phistate_a100_base_", dir="/content") as tmp:
        lab = checkout(Path(tmp))
        unpack_scripts(lab, modules)
        print("Starting | base", PIN, "| experiment", PACKAGE_SHA[:16], "| report", REPORT, flush=True)
        if run_worker(lab):
            print("Worker stopped. Reconnect to A100 and rerun to resume.", flush=True)
            return
    if not REPORT.is_file():
        raise RuntimeError(f"Worker ended without its expected Drive report: {REPORT}")
    status = json.loads(REPORT.read_text(encoding="utf-8"))
    print(f"Drive report: {REPORT}", flush=True)
    for key in ("status", "outcome", "conclusion", "next_action"):
        if key in status:
            print(f"{key}: {status[key]}", flush=True)

    if RELEASE_GPU_AFTER_FINISH and status.get("status") == "complete":
        print("Verified completed Drive report. Releasing this Colab runtime.", flush=True)
        try:
            from google.colab import runtime
            runtime.unassign()
        except Exception as exc:
            print(f"Use Runtime → Disconnect and delete runtime to release the GPU ({exc}).", flush=True)

# Compressed UTF-8 experiment sources; verified before running.
PACKAGE_SHA = "__SHA__"
PREREQUISITE_SOURCE_HASH = "__RECEIPT_SHA__"
PREREQUISITE_QUALIFIED = __QUALIFIED__
SOURCES = (
__SOURCES__
)
launch()
'''


def build(source_dir, output, module_names=None):
    source_dir, output = Path(source_dir), Path(output)
    names = sorted(REQUIRED) if module_names is None else list(module_names)
    if not REQUIRED.issubset(names) or len(names) != len(set(names)) or not all(MODULE.fullmatch(n) for n in names):
        raise ValueError("Select distinct Python basenames including every required experiment module.")
    files = {}
    for name in sorted(names):
        source = (source_dir / name).read_bytes().decode("utf-8")
        compile(source, name, "exec")
        files[name] = source
    receipt_source = files["phistate_base_prerequisite.py"]
    definitions = [node for node in ast.parse(receipt_source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "EVIDENCE" for target in node.targets)]
    if len(definitions) != 1 or not isinstance(definitions[0].value, ast.Call):
        raise ValueError("Expected one embedded evidence receipt.")
    receipt = json.loads(ast.literal_eval(definitions[0].value.args[0]))
    serialized = json.dumps(receipt["payload"], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    qualified = receipt["payload"]["qualified"]
    if (type(qualified) is not bool or hashlib.sha256(serialized).hexdigest() != receipt["sha256"]
            or receipt["payload"]["reference_module_sha256"] != hashlib.sha256(files["phistate_base_reference.py"].encode()).hexdigest()):
        raise ValueError("Invalid reference decision, content hash, or reference implementation.")
    packed = json.dumps(files, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    payload = base64.b64encode(zlib.compress(packed, level=9)).decode("ascii")
    code = (LAUNCHER.replace("__PIN__", PIN).replace("__SHA__", hashlib.sha256(packed).hexdigest())
            .replace("__SOURCES__", "\n".join("    " + repr(part) for part in textwrap.wrap(payload, 100))))
    code = code.replace("__RECEIPT_SHA__", hashlib.sha256(receipt_source.encode()).hexdigest()).replace("__QUALIFIED__", repr(qualified))
    compile(code, "PhiState A100 base comparison Colab launcher", "exec")
    intro = INTRO if qualified else """# PhiState base comparison · qualification pending

**Leave the A100 disconnected.** The archived reference decision for this snapshot is incomplete. Running the cell exits before Drive access, CUDA initialization, or training. A new qualified snapshot is needed before opening an A100 runtime."""

    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata":{"colab": {"name": NAME, "provenance": [], "gpuType": "A100"}, "accelerator":"GPU", "kernelspec":{"display_name":"Python 3","name":"python3"},
                      "language_info":{"name":"python"}},
          "cells":[{"cell_type":"markdown","id":"base-overview","metadata":{},"source":intro.splitlines(keepends=True)},
                   {"cell_type":"code","id":"base-run","metadata":{},"execution_count":None,"outputs":[],
                    "source":code.splitlines(keepends=True)}]}
    if not qualified:
        nb["metadata"].pop("accelerator")
        nb["metadata"]["colab"].pop("gpuType")
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
