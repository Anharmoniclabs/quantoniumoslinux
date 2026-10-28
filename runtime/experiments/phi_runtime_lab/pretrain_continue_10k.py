#!/usr/bin/env python3
"""Continue an EXISTING PhiState-MoE pretraining run without resetting its weights.

Reads all original architecture/optimizer/sampling args from checkpoint train_args.
Uses the exact original prepared corpus and tokenizer; verifies fingerprint.
A new output prefix protects the 5k model. This is not a chat fine-tuner.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import moe_train


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True,
                   help="Existing .last.pt (preferred over best-only .pt)")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out", required=True,
                   help="NEW checkpoint path, never overwrite original")
    p.add_argument("--target-steps", type=int, default=10000,
                   help="TOTAL optimizer steps, not additional steps")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=16)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-seconds", type=float, default=0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-save-optimizer", action="store_true",
                   help="Reduce checkpoint size but later resumes will reset optimizer")
    a = p.parse_args()
    if a.target_steps <= 0 or a.eval_every <= 0 or a.eval_batches <= 0:
        p.error("target-steps, eval-every and eval-batches must be positive")
    source = Path(a.checkpoint).expanduser().resolve()
    dest = Path(a.out).expanduser().resolve()
    if not source.is_file(): p.error("Source checkpoint not found: "+str(source))
    if source == dest: p.error("Output would overwrite input checkpoint")
    if ".last.pt" in dest.name and dest == source:
        p.error("Source and destination must differ")
    _, saved = moe_train.ckpt_load(str(source))
    original = saved.get("train_args")
    if not isinstance(original, dict):
        p.error("Source checkpoint missing train_args. Cannot guarantee compatible continuation.")
    current = int(saved.get("step", -1))
    if current < 0: p.error("Invalid source checkpoint step")
    if current >= a.target_steps:
        print(f"ALREADY AT {current} steps; target {a.target_steps}. No training launched.")
        return
    manifest, _, _ = moe_train.load_data(a.data_dir)
    if saved.get("data_fingerprint") != manifest.get("data_fingerprint"):
        p.error("Corpus fingerprint mismatch. Refusing to resume with new tokenizer/data.")
    if saved.get("tokenizer") != manifest.get("tokenizer"):
        p.error("Tokenizer type mismatch. Refusing.")
    if saved["config"]["vocab"] != manifest["vocab"]:
        p.error("Vocabulary size mismatch.")
    target_last = dest.with_name(dest.stem+".last"+dest.suffix)
    if (dest.exists() or target_last.exists()) and dest != source:
        p.error("Destination already exists. Supply an unused --out or explicitly resume from its .last.pt.")
    args = SimpleNamespace(**original)
    args.action = "train"
    args.data_dir = str(Path(a.data_dir).expanduser().resolve())
    args.resume = str(source)
    args.out = str(dest)
    args.steps = a.target_steps
    args.eval_every = a.eval_every
    args.eval_batches = a.eval_batches
    args.threads = a.threads
    args.max_seconds = a.max_seconds
    args.save_optimizer = not a.no_save_optimizer
    args.dry_run = False
    report = {
        "from_step": current, "to_step": a.target_steps,
        "input": str(source), "output_best": str(dest), "output_latest": str(target_last),
        "corpus_sha256":saved["data_fingerprint"],"tokenizer":saved.get("tokenizer"),
        "optimizer_state_available": "optimizer_state" in saved,
        "optimizer_will_reset": "optimizer_state" not in saved,
        "save_optimizer_in_new_latest":args.save_optimizer,
        "architecture":saved["config"],
        "training_hyperparameters":{k:getattr(args,k,None) for k in
          ("profile","mode","mix_basis","context","batch","accum","lr","optimizer","code_share","seed")}
    }
    print("PLAN",json.dumps(report,indent=2),flush=True)
    if "optimizer_state" not in saved:
        print("NOTE: The older checkpoint has no optimizer state; weights and RNG resume, but optimizer starts fresh.",flush=True)
    if a.dry_run:
        print("DRY RUN OK; no model weights were changed.",flush=True)
        return
    moe_train.train(args)


if __name__ == "__main__":
    main()
