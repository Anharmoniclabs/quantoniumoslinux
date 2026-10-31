"""One bounded optimizer comparison, held-out qualification and native export.

Existing Drive runs are inputs only. A completed experiment never extends itself.
Passing the small qualification suite is not proof of broad assistant competence.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from chat_recovery_data import sha
from chat_recovery_train import atomic_json, run_lock

VERSION = "phistate-coherence-execution-v1"
TERMINAL = {"qualified", "failed_quality", "failed_final"}


def code_identity():
    lab = Path(__file__).resolve().parent
    files = ("coherence_recovery.py", "coherence_recovery_train.py", "coherence_quality.py",
             "optimizer_policy.py", "device_preflight.py", "moe_model.py", "phistate.py",
             "moe_train.py", "chat_recovery_train.py", "chat_recovery_data.py")
    return {name: sha(lab / name) for name in files}


def copy_verified(source, destination, expected):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if sha(destination) != expected:
            raise RuntimeError("Existing immutable copy differs: " + str(destination))
        return
    pending = destination.with_suffix(destination.suffix + ".partial")
    shutil.copyfile(source, pending)
    if sha(pending) != expected or sha(source) != expected:
        raise RuntimeError("Source changed or checkpoint copy failed")
    pending.replace(destination)


def load_model(checkpoint, tokenizer_path, device):
    import sentencepiece as spm
    import torch
    from moe_model import MoEConfig, PhiStateMoELM
    expected = sha(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if sha(checkpoint) != expected or state["tokenizer_sha256"] != sha(tokenizer_path):
        raise RuntimeError("Checkpoint changed or tokenizer mismatch")
    model = PhiStateMoELM(MoEConfig(**state["config"]))
    model.load_state_dict(state["state_dict"], strict=True)
    model.to(device).eval()
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return model, tokenizer


def release_device():
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def choose_candidate(arms):
    """Choose by development results only; never inspect final answers here."""
    eligible = [r for r in arms if r.get("candidate_exists") and r.get("qualified")]
    if not eligible:
        return None
    return max(eligible, key=lambda r: (r["macro_score"], -r["answer_bpt"], r["policy"] == "protected"))


def run(root, output=None, steps=500, device="cuda", eval_every=250, save_every=50, clear_stop=False):
    import torch
    import coherence_quality as quality
    import coherence_recovery_train as trainer
    from device_preflight import run_preflight
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "coherence_execution_v1"
    if not 1 <= steps <= 2000 or eval_every < 1 or save_every < 1:
        raise ValueError("Fixed budget must be 1..2000 updates per arm; intervals must be positive")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime before training")
    torch.set_num_threads(2)
    source = root / "broad_continuation_v1/latest.pt"
    data = root / "corpus_repaired_v1"
    sft = root / "chat_recovery_v1/data"
    forbidden = {source.parent, data, sft, root}
    if output in forbidden or any(parent in forbidden - {root} for parent in output.parents):
        raise RuntimeError("Use a separate experiment output directory")
    with run_lock(output / "execution"):
        with run_lock(source.parent):
            source_hash = sha(source)
            request = dict(version=VERSION, source=str(source), source_sha256=source_hash,
                           steps_per_arm=steps, eval_every=eval_every, save_every=save_every,
                           code_sha256=code_identity(), torch_version=str(torch.__version__),
                           sft_manifest_sha256=sha(sft / "manifest.json"),
                           corpus_manifest_sha256=sha(data / "manifest.json"),
                           tokenizer_sha256=sha(data / "tokenizer.model"))
            output.mkdir(parents=True, exist_ok=True)
            request_path = output / "request.json"
            if request_path.exists():
                if json.loads(request_path.read_text()) != request:
                    raise RuntimeError("Experiment source/settings changed; existing budget is immutable")
            else:
                if any(output.iterdir()):
                    raise RuntimeError("Unrecognized nonempty experiment directory")
                atomic_json(request_path, request)
            frozen = output / "source.pt"
            copy_verified(source, frozen, source_hash)
        report_path = output / "report.json"
        if report_path.exists():
            prior = json.loads(report_path.read_text())
            if prior.get("status") in TERMINAL:
                print("EXPERIMENT ALREADY ENDED:", prior["status"], "— no automatic extension", flush=True)
                return prior
        stop = output / "STOP"
        if stop.exists():
            if clear_stop:
                stop.unlink()
            else:
                print("Saved STOP request present; use --clear-stop to resume this fixed budget.", flush=True)
                return {"status": "stopped"}
        preflight = run_preflight(frozen, data / "tokenizer.model", output / "device_preflight.json", device=device)
        if not preflight.get("passed"):
            raise RuntimeError("Device preflight failed; training was not started. See device_preflight.json")
        release_device()
        quality.build_suites(sft, output / "qualification")
        dev_path = output / "qualification/dev.json"
        final_path = output / "qualification/final.json"
        arms = []
        for policy in ("original", "protected"):
            if stop.exists():
                return {"status": "stopped", "arms": arms}
            arm = output / policy
            args = SimpleNamespace(checkpoint=str(frozen), data_dir=str(data), sft_dir=str(sft),
                run_dir=str(arm), steps=steps, policy=policy, device=device,
                eval_every=eval_every, save_every=save_every, dev_suite=str(dev_path),
                clear_stop=clear_stop, parent_stop=str(stop))
            print("\nARM:", policy, "| at most", steps, "additional updates", flush=True)
            progress = trainer.run(args)
            status = progress["status"]
            release_device()
            if status == "stopped":
                return {"status": "stopped", "arms": arms}
            progress = json.loads((arm / "progress.json").read_text())
            candidate = arm / "candidate.pt"
            row = dict(policy=policy, status=status, progress=progress,
                       candidate_exists=candidate.exists(), qualified=False)
            if candidate.exists():
                state = torch.load(candidate, map_location="cpu", weights_only=True)
                # The trainer records the exact development result on this checkpoint.
                dev = state["quality"]
                row.update(qualified=bool(dev["qualified"]), macro_score=float(dev["macro_score"]),
                           answer_bpt=float(state["last_validation"]["val_answer_bpt"]),
                           candidate=str(candidate), candidate_sha256=sha(candidate))
                del state
            arms.append(row)
            atomic_json(report_path, dict(version=VERSION, status="comparing", arms=arms, request=request))
        selected = choose_candidate(arms)
        report = dict(version=VERSION, request=request, arms=arms, source_unchanged=sha(source) == source_hash,
                      scope="Fixed development and final functional checks; not broad assistant certification",
                      native_export=None)
        if not report["source_unchanged"]:
            raise RuntimeError("Source checkpoint changed externally during comparison")
        if selected is None:
            report.update(status="failed_quality", reason="Neither arm qualified on development tasks",
                          next_action="Stop this experiment. Review per-task failures and update scales before a new recipe.")
            atomic_json(report_path, report)
            print("FAILED QUALITY — checkpoints saved; no native candidate promoted.", flush=True)
            return report
        report["selected"] = selected["policy"]
        candidate = Path(selected["candidate"])
        final_cache = output / "final_result.json"
        identity = dict(candidate_sha256=sha(candidate), suite_sha256=sha(final_path))
        if final_cache.exists():
            final = json.loads(final_cache.read_text())
            if final["identity"] != identity:
                raise RuntimeError("Final test was already used for a different candidate; do not retune on it")
        else:
            model, tokenizer = load_model(candidate, data / "tokenizer.model", device)
            final = dict(identity=identity, result=quality.evaluate_suite(model, tokenizer, quality.load_suite(final_path)))
            del model, tokenizer
            release_device()
            atomic_json(final_cache, final)
        report["final"] = final
        if not final["result"]["qualified"]:
            report.update(status="failed_final", reason="Selected candidate failed untouched final checks")
            atomic_json(report_path, report)
            print("FAILED FINAL CHECK — no native candidate promoted.", flush=True)
            return report
        native = output / "qualified_native"
        native.mkdir(exist_ok=True)
        copy_verified(candidate, native / "best.pt", identity["candidate_sha256"])
        try:
            from chat_recovery_cpp import export
            exported = export(native, sft)
            report.update(status="qualified", native_export=str(exported), native_run=str(native))
        except Exception as error:
            report.update(status="native_export_failed", reason=str(error))
            atomic_json(report_path, report)
            raise
        atomic_json(report_path, report)
        print("QUALIFIED ON FIXED CHECKS; native parity passed:", native, flush=True)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True)
    p.add_argument("--output")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--clear-stop", action="store_true")
    args = p.parse_args()
    result = run(**vars(args))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
