# Run from a Colab bootstrap that defines REPAIR_CODE_COMMIT, LAB and DRIVE.
# This cell STARTS a bounded GPU continuation; re-running resumes latest.pt.
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

assert torch.cuda.is_available(), "This continuation cell requires the connected Colab GPU."
repo = Path(LAB).parents[2]
runner = Path("/content/phistate_whitespace_runner") / REPAIR_CODE_COMMIT
runner.mkdir(parents=True, exist_ok=True)
for filename in (
    "continue_whitespace_repair.py", "repair_whitespace_corpus.py",
    "moe_train.py", "moe_model.py", "phistate.py",
):
    relative = f"runtime/experiments/phi_runtime_lab/{filename}"
    content = subprocess.check_output([
        "git", "-C", str(repo), "show", f"{REPAIR_CODE_COMMIT}:{relative}",
    ])
    (runner / filename).write_bytes(content)

repair = Path(DRIVE) / "whitespace_repair_v1"
run_dir = repair / "continue_60k"
run_dir.mkdir(parents=True, exist_ok=True)
source_data = Path(globals().get("EXPANDED_DATA", Path(DRIVE) / "data/hf_coding_knowledge_v2_expanded"))
TARGET_TOTAL_STEPS = 61000
command = [
    sys.executable, "-u", str(runner / "continue_whitespace_repair.py"),
    "--checkpoint", str(Path(DRIVE) / "50k/phi_60000.last.pt"),
    "--source-data", str(source_data),
    "--data-dir", str(repair / "corpus_repaired_v1"),
    "--run-dir", str(run_dir),
    "--target-steps", str(TARGET_TOTAL_STEPS),
    "--save-every", "100", "--eval-batches", "16",
    "--threads", "2", "--device", "cuda", "--clear-stop",
]
log_path = run_dir / f"run_{time.time_ns()}.log"
print("CONTINUING saved 60K model toward", TARGET_TOTAL_STEPS, "total steps", flush=True)
print("Saved progress:", run_dir / "progress.json", flush=True)
print("Safe stop: create", run_dir / "STOP", flush=True)
print("Log:", log_path, flush=True)
with log_path.open("w", encoding="utf-8") as log:
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, bufsize=1) as process:
        def show_output():
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
        try:
            show_output()
        except KeyboardInterrupt:
            print("Requesting a checkpoint after the current optimizer step...", flush=True)
            if process.poll() is None:
                process.terminate()
            show_output()
        status = process.wait()
if status:
    raise RuntimeError(f"Continuation stopped with an error; inspect {log_path}. Prior saved checkpoints remain available.")
progress_path = run_dir / "progress.json"
if progress_path.is_file():
    print(json.dumps(json.loads(progress_path.read_text()), indent=2))
