#!/usr/bin/env python3
"""Restart-safe CUDA pretraining continuation to 50,000 TOTAL optimizer steps.

Runs existing PhiState-MoE continuation in bounded segments, then commits a
verified best/latest checkpoint pair to alternating Google Drive slots.
The progress marker is updated only after both files are persisted and valid.
If Colab disconnects, re-run with the same Drive folder to resume from the
last completed segment, without overwriting the 10,250-step source checkpoint.
This is an attended Colab session: it cannot run while the runtime is stopped.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import torch


def load_checkpoint(path, fingerprint):
    item = torch.load(path, map_location="cpu", weights_only=True)
    if item.get("format") != "PhiStateMoE-v1":
        raise RuntimeError(f"Wrong checkpoint format: {path}")
    if item.get("data_fingerprint") != fingerprint:
        raise RuntimeError(f"Corpus fingerprint mismatch: {path}")
    if not isinstance(item.get("step"), int):
        raise RuntimeError(f"Checkpoint is missing its step: {path}")
    return item


def verify_pair(latest, best, fingerprint):
    last_ck = load_checkpoint(latest, fingerprint)
    best_ck = load_checkpoint(best, fingerprint)
    if "optimizer_state" not in last_ck:
        raise RuntimeError(f"Resumable latest checkpoint lacks optimizer state: {latest}")
    if best_ck["config"] != last_ck["config"]:
        raise RuntimeError("Best/latest architecture mismatch")
    if best_ck["step"] > last_ck["step"]:
        raise RuntimeError("Best checkpoint is newer than latest")
    if best_ck.get("best_validation_bpt", 1e9) > last_ck.get("best_validation_bpt", 1e9) + 1e-5:
        raise RuntimeError("Best/latest checkpoint validation metadata mismatch")
    return int(last_ck["step"]), float(last_ck.get("best_validation_bpt", float("inf")))


def atomic_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    pending = destination.with_name(destination.name + ".pending")
    try:
        with source.open("rb") as src, pending.open("wb") as dst:
            shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        if pending.stat().st_size != source.stat().st_size:
            raise IOError(f"Incomplete copy: {destination}")
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


def atomic_json(path, data):
    pending = path.with_name(path.name + ".pending")
    try:
        with pending.open("w", encoding="utf8") as f:
            json.dump(data, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def run_one_chunk(lab, data, latest, out, target, threads, eval_batches, logfile):
    cmd = [
        sys.executable, "-u", str(lab / "pretrain_continue_10k.py"),
        "--checkpoint", str(latest), "--data-dir", str(data),
        "--out", str(out), "--target-steps", str(target),
        "--eval-every", str(target), "--eval-batches", str(eval_batches),
        "--threads", str(threads), "--device", "cuda",
    ]
    print("TRAINING:", " ".join(cmd), flush=True)
    result = None
    with logfile.open("w", encoding="utf8") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            if line.startswith("RESULT "):
                result = json.loads(line[7:])
        code = process.wait()
    if code:
        raise RuntimeError(f"Training stopped with exit {code}; inspect the preceding output or {logfile}")
    if result is None or result["step"] != target:
        raise RuntimeError(f"No evaluation result for step {target}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lab", required=True, help="PhiState Python source directory")
    p.add_argument("--data-dir", required=True, help="Original local HF V2 corpus")
    p.add_argument("--initial-last", required=True, help="Saved step-10,250 optimizer checkpoint in Drive")
    p.add_argument("--initial-best", required=True, help="Matching best checkpoint in Drive")
    p.add_argument("--drive-dir", required=True, help="Private, mounted Google Drive checkpoint folder")
    p.add_argument("--work-dir", default="/content/phistate_50k_work")
    p.add_argument("--target-steps", type=int, default=50000)
    p.add_argument("--chunk-steps", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--max-chunks", type=int, default=0,
                   help="0 = train through target; 1 = run one safe segment and exit")
    a = p.parse_args()
    if a.target_steps < 10250 or min(a.chunk_steps, a.threads, a.eval_batches) < 1 or a.max_chunks < 0:
        p.error("Invalid step/chunk/thread settings")
    if not torch.cuda.is_available():
        p.error("A CUDA GPU is required; change the Colab runtime type")
    lab = Path(a.lab).expanduser().resolve()
    data = Path(a.data_dir).expanduser().resolve()
    initial_latest = Path(a.initial_last).expanduser().resolve()
    initial_best = Path(a.initial_best).expanduser().resolve()
    drive = Path(a.drive_dir).expanduser().resolve()
    drive.mkdir(parents=True, exist_ok=True)
    work = Path(a.work_dir).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    for item in (lab / "pretrain_continue_10k.py", data / "manifest.json", initial_latest, initial_best):
        if not item.is_file():
            raise FileNotFoundError(f"Required source not found: {item}")
    manifest = json.loads((data / "manifest.json").read_text())
    fingerprint = manifest["data_fingerprint"]
    init_step, init_best = verify_pair(initial_latest, initial_best, fingerprint)
    if init_step != 10250:
        raise RuntimeError(f"Expected initial Drive checkpoint step 10250, found {init_step}")
    marker = drive / "progress.json"
    active = None
    if marker.is_file():
        state = json.loads(marker.read_text())
        active = state.get("active_slot")
        if active not in ("a", "b") or state.get("data_fingerprint") != fingerprint:
            raise RuntimeError("Existing progress marker does not match this run")
        source_latest = drive / f"slot_{active}.last.pt"
        source_best = drive / f"slot_{active}.pt"
        current, score = verify_pair(source_latest, source_best, fingerprint)
        if current != state["step"]:
            raise RuntimeError("Progress marker and checkpoint step disagree")
        if current < init_step:
            raise RuntimeError("Progress marker is older than original source")
        print(f"RESUMED FROM GOOGLE DRIVE: step {current}", flush=True)
    else:
        source_latest, source_best = initial_latest, initial_best
        current, score = init_step, init_best
        print("STARTING FROM PROTECTED 10,250-STEP CHECKPOINT", flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("CORPUS:", fingerprint, "TARGET:", a.target_steps,
          "CHUNK:", a.chunk_steps, "CURRENT:", current, flush=True)
    if current >= a.target_steps:
        print("TARGET ALREADY REACHED:", current, flush=True)
        return

    milestones = {20000, 30000, 40000, 50000}
    chunks = 0
    with tempfile.TemporaryDirectory(prefix="segments_", dir=str(work)) as temporary:
        tmp = Path(temporary)
        source_local_last = tmp / "start.last.pt"
        source_local_best = tmp / "start.pt"
        shutil.copy2(source_latest, source_local_last)
        shutil.copy2(source_best, source_local_best)
        verify_pair(source_local_last, source_local_best, fingerprint)

        while current < a.target_steps:
            if a.max_chunks and chunks >= a.max_chunks:
                print("PAUSED AT COMPLETED BACKUP STEP:", current, flush=True)
                break
            next_milestones = [m for m in milestones if current < m <= a.target_steps]
            boundary = min(next_milestones) if next_milestones else a.target_steps
            end = min(current + a.chunk_steps, a.target_steps, boundary)
            out = tmp / f"segment_{end}.pt"
            finished_last = tmp / f"segment_{end}.last.pt"
            logfile = tmp / f"segment_{end}.log"
            result = run_one_chunk(lab, data, source_local_last, out,
                                   end, a.threads, a.eval_batches, logfile)
            step, new_best = verify_pair(finished_last, out, fingerprint)
            if step != end:
                raise RuntimeError("Latest checkpoint did not reach expected segment end")
            # Two slots: complete the inactive backup before changing the pointer.
            new_active = "b" if active == "a" else "a"
            dest_last = drive / f"slot_{new_active}.last.pt"
            dest_best = drive / f"slot_{new_active}.pt"
            print(f"BACKING UP: step {step} -> Drive slot {new_active}", flush=True)
            atomic_copy(finished_last, dest_last)
            atomic_copy(out, dest_best)
            backup_step, backup_best = verify_pair(dest_last, dest_best, fingerprint)
            if backup_step != step or abs(backup_best - new_best) > 1e-5:
                raise RuntimeError("New Google Drive backup verification failed")
            state = {
                "format": "PhiStateMoE-L4-50K-progress-v1",
                "step": step, "active_slot": new_active,
                "best_validation_bpt": backup_best,
                "data_fingerprint": fingerprint,
                "target_steps": a.target_steps,
                "chunk_steps": a.chunk_steps,
                "updated_utc": datetime.now(timezone.utc).isoformat(),
                "last_session_tokens_per_second": result.get("session_tokens_per_second"),
                "last_val_mean_bpt": result["val_mean_bpt"],
            }
            atomic_json(marker, state)
            if step in milestones:
                milestone_dir = drive / "milestones"
                print(f"PRESERVING MILESTONE {step} in Drive", flush=True)
                atomic_copy(finished_last, milestone_dir / f"phi_{step}.last.pt")
                atomic_copy(out, milestone_dir / f"phi_{step}.pt")
                milestone_step, _ = verify_pair(milestone_dir / f"phi_{step}.last.pt",
                                                milestone_dir / f"phi_{step}.pt", fingerprint)
                if milestone_step != step:
                    raise RuntimeError("Milestone checkpoint verification failed")
            try:
                atomic_copy(logfile, drive / "latest_segment.log")
                with (drive / "history.jsonl").open("a", encoding="utf8") as history:
                    history.write(json.dumps(result, sort_keys=True) + "\n")
                    history.flush()
            except OSError as warning:
                print("WARNING: optional history/log backup failed, but checkpoint + progress marker are safe:",
                      warning, flush=True)
            print("PERSISTED & VERIFIED:", step,
                  "validation BPT:", result["val_mean_bpt"],
                  "session tok/s:", result.get("session_tokens_per_second"),
                  flush=True)
            # The older local source can now be released; the new checkpoint
            # is in Google Drive even if the runtime disconnects immediately.
            source_local_last.unlink(missing_ok=True)
            source_local_best.unlink(missing_ok=True)
            source_local_last, source_local_best = finished_last, out
            current, score, active = step, backup_best, new_active
            chunks += 1
    print("CURRENT SAVED STEP:", current, "/", a.target_steps, flush=True)
    print("GOOGLE DRIVE PROGRESS:", marker, flush=True)
    if current >= a.target_steps:
        print(f"{a.target_steps:,} OPTIMIZER STEPS FINISHED AND BACKED UP", flush=True)
    else:
        print("Rerun this command after reconnecting to continue from Drive", flush=True)


if __name__ == "__main__":
    main()
