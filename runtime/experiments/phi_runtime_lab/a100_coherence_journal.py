"""Pinned, bounded A100 coherence experiment descending from the archived SFT-4000 checkpoint."""

import argparse
import json
import os
import time
from pathlib import Path

import sentencepiece as spm
import torch

import a100_dialogue_checks as dialogue
import a100_foundation_control
import coherence_quality
import data_comparison_train
import data_mixture_comparison as plans
import paired_instruction_v2
import paired_quality_v2 as quality
import phistate_a100_diagnostics
from a100_run_guard import clean_orphans, guarded_run
from chat_recovery_data import TEMPLATE, sha, verify_prepared
from chat_recovery_train import atomic_json, load_rows
from coherence_recovery import copy_verified
from device_preflight import run_preflight
from moe_train import ckpt_load

OUTPUT = "phistate_a100_coherence_v2"
PARENT_SHA = "dcc79a5a3b05411725fefcefa04fd0aafd4393f4efb21e018c60da1e9a80d8ea"
STEPS = 256
TARGETS = 128
LAB = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bound_result(path, identity, measure):
    """Publish complete measurements atomically and reuse only the identical evaluation."""
    if path.is_file():
        record = read(path)
        if record["identity"] != identity or plans.digest(record["result"]) != record["result_sha256"]:
            raise RuntimeError(f"Saved evaluation has another identity or changed contents: {path}")
        return record["result"]
    result = measure()
    atomic_json(path, {"identity": identity, "result": result, "result_sha256": plans.digest(result)})
    return result


def source_request(root, out):
    prior = root / "coherence_execution_v1"
    parent, data = prior / "source.pt", root / "chat_recovery_v1/data"
    previous = read(prior / "report.json")
    if (
        previous.get("status") != "failed_quality"
        or previous["request"]["source_sha256"] != PARENT_SHA
        or sha(parent) != PARENT_SHA
    ):
        raise ValueError("Expected the archived failed-quality comparison and its exact SFT-4000 source")
    code = (
        "a100_coherence_journal.py",
        "a100_run_guard.py",
        "a100_foundation_control.py",
        "a100_greedy_batch.py",
        "a100_dialogue_checks.py",
        "paired_instruction_v2.py",
        "paired_quality_v2.py",
        "phistate_a100_diagnostics.py",
        "data_mixture_comparison.py",
        "data_comparison_train.py",
        "coherence_quality.py",
        "chat_recovery_train.py",
        "chat_recovery_data.py",
        "coherence_recovery.py",
        "coherence_recovery_train.py",
        "device_preflight.py",
        "prompt_conditioning_eval.py",
        "generalization_train.py",
        "paired_instruction_data.py",
        "chat_recovery_cpp.py",
        "native_cpp_trial.py",
        "moe_train.py",
        "moe_model.py",
        "phistate.py",
    )
    native_src = LAB.parents[1] / "src/rftmw_native"
    request = {
        "version": "PhiState-A100-coherence-v2",
        "source_sha256": PARENT_SHA,
        "source_sft_step": 4000,
        "steps_per_arm": STEPS,
        "answer_targets_per_update": TARGETS,
        "common_targets_per_update": TARGETS // 2,
        "dialogue_min_pair_rate": 0.5,
        "seed": plans.SEED,
        "replay": 0,
        "optimizer": "parent Adafactor",
        "dtype": "FP32",
        "hardware": "A100",
        "tokenizer_sha256": sha(data / "tokenizer.model"),
        "data_manifest_sha256": sha(data / "manifest.json"),
        "prior_report_sha256": sha(prior / "report.json"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "sentencepiece_version": spm.__version__,
        "commit": os.environ.get("PHISTATE_JOURNAL_COMMIT"),
        "code": {p: sha(LAB / p) for p in code},
        "native_code": {
            p.name: sha(p) for p in [native_src / "phistate_moe_cli.cpp", *sorted(native_src.glob("*.hpp"))]
        },
    }
    if not (out / "request.json").exists() and any(path.name != "STOP" for path in out.iterdir()):
        raise RuntimeError(f"Unidentified files in new output directory: {out}")
    plans._fixed_json(out / "request.json", request)
    return request, parent, data, prior


def verify_source(parent, data, request):
    manifest = read(data / "manifest.json")
    prepared = verify_prepared(data, request["tokenizer_sha256"], manifest["context"])
    state = torch.load(parent, map_location="cpu", weights_only=True)
    if not (
        state.get("training_stage") == "chat_recovery_v1"
        and state.get("sft_steps") == 4000
        and state.get("pretrain_step") == 65000
        and state.get("chat_template") == TEMPLATE
        and state.get("tokenizer_sha256") == request["tokenizer_sha256"]
        and state.get("sft_data_fingerprint") == prepared["data_fingerprint"]
        and state.get("data_fingerprint") == prepared["base_data_fingerprint"]
        and state.get("train_args", {}).get("optimizer") == "adafactor"
        and "optimizer_state" in state
    ):
        raise RuntimeError("Chat lineage, template, tokenizer, or optimizer verification failed")
    return prepared


def suites(data, pools, out):
    # Exclude every broad user turn, including earlier turns of multi-turn training examples.
    forbidden = {
        coherence_quality.normalized(m["content"])
        for split in pools.values()
        for bucket in split.values()
        for row in bucket
        for m in row["messages"]
        if m["role"] == "user"
    }
    fixed = {}
    provenance = {f: sha(data / f) for f in ("train.jsonl", "val.jsonl")}
    suite_dir = out / "broad_suites"
    suite_dir.mkdir(exist_ok=True)
    for split in ("dev", "final"):
        items = coherence_quality._bank(split)
        for item in items:
            choices = [
                item["prompt"],
                "Please answer this request: " + item["prompt"],
                "Complete this task: " + item["prompt"],
            ]
            prompt = next((p for p in choices if coherence_quality.normalized(p) not in forbidden), None)
            if prompt is None:
                raise ValueError("Broad prompt alternatives overlap prepared user turns: " + item["id"])
            item["prompt"] = prompt
            forbidden.add(coherence_quality.normalized(prompt))
        suite = {
            "format": coherence_quality.FORMAT,
            "split": split,
            "scope": coherence_quality.SCOPE,
            "criteria": coherence_quality.CRITERIA,
            "prepared_files": provenance,
            "items": items,
        }
        suite["fingerprint"] = coherence_quality.fingerprint(suite)
        target = suite_dir / (split + ".json")
        plans._fixed_json(target, suite)
        fixed[split] = coherence_quality.load_suite(target)
    return fixed, forbidden


def decide(out, candidate, screen, latest, tokenizer, paired_final, broad_final, device, guard):
    """Evaluate a single development-selected candidate, persist each sealed phase, then export."""
    broad_dev = candidate["broad"]
    if not screen["eligible"]:
        return quality.final_decision(screen, broad_dev, None), None
    identity = {
        "checkpoint": sha(latest),
        "paired_suite": plans.digest(paired_final),
        "broad_suite": broad_final["fingerprint"],
        "dialogue_suite": plans.digest(dialogue.bank("final")),
        "grader": quality.GRADER_SHA,
        "broad_dev_qualified": broad_dev["qualified"],
    }
    plans._fixed_json(out / "selected_candidate.json", identity)
    model = None

    def evaluate(fn, items):
        nonlocal model
        guard.check()
        if model is None:
            model, state = ckpt_load(str(latest))
            del state
            if sha(latest) != identity["checkpoint"]:
                raise RuntimeError("Selected checkpoint changed")
            model.to(device).eval()
        value = fn(model, tokenizer, items)
        guard.check()
        return value

    pair_scores = bound_result(
        out / "sealed_pairs.json", identity, lambda: evaluate(quality.evaluate_pairs, paired_final)
    )
    decision = quality.final_decision(screen, broad_dev, pair_scores)
    if decision["status"] == "broad_final_required" and candidate["dialogue"]["pair_rate"] < 0.5:
        decision = {
            "status": "paired_transfer_only",
            "paired_passed": True,
            "eligible_for_native_export": False,
            "reason": "Multi-turn development passed fewer than half of changed-history pairs",
        }
    if decision["status"] == "broad_final_required":
        broad_scores = bound_result(
            out / "sealed_broad.json", identity, lambda: evaluate(quality.evaluate_suite, broad_final)
        )
        decision = quality.final_decision(screen, broad_dev, pair_scores, broad_scores)
        if decision["eligible_for_native_export"]:
            conversation = bound_result(
                out / "sealed_dialogue.json", identity, lambda: evaluate(dialogue.evaluate, "final")
            )
            if conversation["pair_rate"] < 0.5:
                decision = {
                    "status": "broad_tasks_only",
                    "paired_passed": True,
                    "eligible_for_native_export": False,
                    "reason": "Multi-turn final passed fewer than half of changed-history pairs",
                }
    del model
    plans._release()
    return decision, identity


def run(root, recover=False):
    root = Path(root).resolve()
    out = root / OUTPUT
    if not root.is_dir():
        raise FileNotFoundError(f"Project not mounted at {root}")
    if not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError("Choose an A100 GPU runtime in Colab before running this journal")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with guarded_run(out, recover=recover) as guard:
        out.mkdir(exist_ok=True)
        clean_orphans(out)
        request, parent, data, prior = source_request(root, out)
        report_path = out / "report.json"
        if report_path.is_file():
            existing = read(report_path)
            if existing.get("status") == "complete" and existing.get("request") == request:
                print("Already complete:", existing["outcome"], "|", report_path, flush=True)
                return existing

        def publish(status, phase, **fields):
            guard.check()
            info = dict(status=status, phase=phase, request=request, **fields)
            atomic_json(report_path, info)
            print(status.upper(), phase, "|", report_path, flush=True)
            return info

        stop = out / "STOP"

        def stopped(phase):
            return publish(
                "stopped", phase, resume="Remove the intentional STOP file before resuming, if present."
            )

        if stop.exists():
            return stopped("before preflight")
        phase = "source verification"
        try:
            publish("running", phase)
            prepared = verify_source(parent, data, request)
            # These earlier experiments use the same broad final bank. Reuse would contaminate selection.
            for final in (
                prior / "final_result.json",
                root / "paired_data_comparison_v1/final_result.json",
            ):
                if final.is_file():
                    raise RuntimeError(f"The legacy broad final may already have been evaluated: {final}")
            tokenizer_path = data / "tokenizer.model"
            phase = "device and frozen-source diagnostics"
            publish("running", phase)
            preflight = run_preflight(parent, tokenizer_path, out / "device_preflight.json", device="cuda")
            if not preflight["passed"]:
                raise RuntimeError(f"CUDA preflight failed. See {out / 'device_preflight.json'}")
            diagnostics = phistate_a100_diagnostics.run(
                parent,
                data,
                out / "source_diagnostics",
                device="cuda",
                conditioning_dir=prior / "prompt_conditioning",
            )
            if diagnostics["status"] != "complete" or diagnostics["optimizer_steps_taken"] != 0:
                raise RuntimeError("Frozen-source diagnostic did not complete")
            tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
            pools = load_rows(data, tokenizer, prepared["context"])
            broad, forbidden = suites(data, pools, out)
            pairs = paired_instruction_v2.build(tokenizer, prepared["context"], excluded_prompts=forbidden)
            plan = plans.schedules(pools["train"], pairs["train"], STEPS, TARGETS)
            summaries = {name: plans.describe(rows) for name, rows in plan.items()}
            data_id = {
                "arms": summaries,
                "train_rows": len(pairs["train"]),
                "paired_dev": plans.digest(pairs["dev"]),
                "paired_final": plans.digest(pairs["final"]),
                "broad_dev": broad["dev"]["fingerprint"],
                "broad_final": broad["final"]["fingerprint"],
                "dialogue_dev": plans.digest(dialogue.bank("dev")),
                "dialogue_final": plans.digest(dialogue.bank("final")),
            }
            plans._fixed_json(out / "data_plan.json", data_id)

            def evaluate(model):
                guard.check()
                result = quality.evaluate_dev(model, tokenizer, pairs["dev"], broad["dev"], pools["val"])
                result["dialogue"] = dialogue.evaluate(model, tokenizer, "dev")
                return result

            arms, timing = {}, {}
            for name in ("broad", "paired"):
                phase = name + " training and development"
                if stop.exists():
                    return stopped(phase)
                publish("running", phase, progress_file=str(out / name / "progress.json"))
                start = time.monotonic()
                torch.cuda.reset_peak_memory_stats()
                info = data_comparison_train.run_arm(
                    parent=parent,
                    tokenizer=tokenizer,
                    pool=plan[name],
                    output=out / name,
                    policy_name=name,
                    device="cuda",
                    steps=STEPS,
                    tokens_per_step=TARGETS,
                    seed=plans.SEED,
                    evaluate_callback=evaluate,
                    identity={
                        "request": request,
                        "schedule": summaries[name]["schedule_sha256"],
                        "paired_dev": data_id["paired_dev"],
                        "broad_dev": data_id["broad_dev"],
                    },
                    stop_path=stop,
                    check_owner=guard.check,
                )
                if info["status"] == "stopped":
                    return stopped(phase)
                if (info["status"], info["phase_step"], info["supervised_tokens"]) != (
                    "complete",
                    STEPS,
                    STEPS * TARGETS,
                ):
                    raise RuntimeError("Incomplete comparison budget")
                timing[name] = {
                    "seconds_this_invocation": time.monotonic() - start,
                    "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                }
                arms[name] = info
                plans._release()
            phase = "development selection"
            publish("running", phase)
            if sha(parent) != PARENT_SHA:
                raise RuntimeError("Frozen source changed during the run")

            def metric(name, index):
                return arms[name]["development_curve"][index]["metrics"]

            baseline, control, candidate = metric("broad", 0), metric("broad", -1), metric("paired", -1)
            if any(
                baseline[k]["rows"] != metric("paired", 0)[k]["rows"]
                for k in ("paired", "broad", "dialogue")
            ):
                raise RuntimeError("Starting models generated different baseline answers")
            screen = quality.screen_candidate(baseline, control, candidate)
            dialog_retained = candidate["dialogue"]["pair_rate"] >= max(
                baseline["dialogue"]["pair_rate"], control["dialogue"]["pair_rate"]
            )
            screen["checks"]["dialogue_pairs_retained"] = dialog_retained
            if not dialog_retained:
                screen["eligible"] = False
                screen["reasons"].append("dialogue_pairs_retained")
            plans._fixed_json(out / "development_screen.json", screen)
            review = quality.blind_review_packet(
                {"parent": baseline["broad"], "control": control["broad"], "candidate": candidate["broad"]}
            )
            for name in ("packet", "key"):
                plans._fixed_json(out / f"blind_review_{name}.json", review[name])
            if stop.exists():
                return stopped(phase)
            phase = "sealed evaluation"
            latest = Path(arms["paired"]["latest"])
            decision, identity = decide(
                out, candidate, screen, latest, tokenizer, pairs["final"], broad["final"], "cuda", guard
            )
            native, fallback = None, None
            if decision["eligible_for_native_export"]:
                phase = "native parity and separate export"
                publish("running", phase, quality_decision=decision)
                from chat_recovery_cpp import export

                native_dir = out / "qualified_native"
                native_dir.mkdir(exist_ok=True)
                copy_verified(latest, native_dir / "best.pt", identity["checkpoint"])
                native = str(export(native_dir, data))
            else:
                phase = "fresh dense and PhiState learning control"
                publish("running", phase, quality_decision=decision)
                fallback = a100_foundation_control.run(
                    tokenizer, out / "foundation_control", device="cuda", stop_path=stop
                )
                if fallback["status"] == "stopped":
                    return stopped(phase)
                if fallback["status"] != "complete":
                    raise RuntimeError("Foundation diagnostic did not finish")
            if sha(parent) != PARENT_SHA:
                raise RuntimeError("Source checkpoint changed before completion")
            scores = {
                name: {
                    "pair_hits": m["paired"]["pairs_passed"],
                    "pairs": m["paired"]["pairs_total"],
                    "broad_hits": sum(b["passed"] for b in m["broad"]["buckets"].values()),
                    "broad_total": sum(b["total"] for b in m["broad"]["buckets"].values()),
                    "dialogue_hits": m["dialogue"]["pairs_passed"],
                    "dialogue_pairs": m["dialogue"]["pairs_total"],
                    "heldout_answer_bpt": m["answer_validation"]["val_answer_bpt"],
                }
                for name, m in [
                    ("parent", baseline),
                    ("broad_control", control),
                    ("paired_candidate", candidate),
                ]
            }
            examples = {}
            for name, metrics in [
                ("parent", baseline),
                ("broad_control", control),
                ("paired_candidate", candidate),
            ]:
                rows = metrics["broad"]["rows"]
                examples[name] = [
                    {"prompt": row["prompt"], "answer": row["text"], "passed": row["passed"]}
                    for row in rows
                    if row["id"].endswith("-0")
                ]
            result = publish(
                "complete",
                "finished",
                outcome=decision["status"],
                development=scores,
                examples=examples,
                development_screen=screen,
                quality_decision=decision,
                native_export=native,
                foundation_outcome=fallback.get("outcome") if fallback else None,
                foundation_report=str(out / "foundation_control/report.json") if fallback else None,
                source_unchanged=True,
                timing=timing,
            )
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in ("outcome", "development", "foundation_outcome", "native_export")
                    },
                    indent=2,
                ),
                flush=True,
            )
            return result
        except Exception as exc:
            publish("error", phase, error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument(
        "--recover-disconnected",
        action="store_true",
        help="After the old runtime ended, recover an abandoned stale owner directory.",
    )
    args = parser.parse_args()
    run(args.root, args.recover_disconnected)
