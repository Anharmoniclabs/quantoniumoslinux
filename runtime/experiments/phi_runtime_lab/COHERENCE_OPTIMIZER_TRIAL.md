# Bounded optimizer comparison after whitespace repair

The 65K checkpoint passed native C++/PyTorch inference parity but generated
whitespace and repetitive text. The 60K–65K phase trained only 2,560,000 token
exposures with the repaired tokenizer and excluded the expanded instruction
additions. Falling aggregate validation loss has not established useful output.

`moe_train.optimizer_for` passes `lr=0.0003` directly to PyTorch Adafactor. In
that optimizer, the relative step is capped by lr, then multiplied by the
parameter tensor RMS (floored at epsilon2). This is not AdamW learning-rate
semantics and warrants a controlled test; it does not prove the sole cause of
the generation failure. Do not blindly increase Adafactor lr on the live run.

`coherence_optimizer_trial.py` starts two independent arms from the same 65K
weights:

- Control: restore the existing Adafactor optimizer and settings.
- Candidate: fresh AdamW, lr 0.0001, linear warmup for 50 steps (or the entire
  trial if shorter), otherwise the existing data, batch, context, accumulation,
  domain share, clipping, weight decay and router losses. AdamW uses PyTorch's
  default betas. This compares optimizer configurations, not only learning rate.

Each arm runs 200 updates by default, using the same seeded batches. A digest
confirms exact batch equality. With the existing context128/batch1/accum4,
each arm processes 102,400 token exposures. Validation uses 64 evenly spaced
windows per domain (16,384 total tokens); these scores are not the original
4,096-token progress.json evaluation. No test split is evaluated.

Validation reports all-token, whitespace-only-token and other-token BPT.
Whitespace classification includes byte fallback newline/tab tokens, correcting
the earlier notebook's undercount. Other tokens include domain and boundary
tokens, so this is a diagnostic split, not a semantic-understanding metric.
Greedy generations use fixed prompts without suppression or repetition penalties.

The program writes two independent candidate checkpoint files, every100 steps
and at completion, plus report.json, into a new output directory. It refuses
to overwrite an existing output directory and verifies the original checkpoint
hash at the end. It does not promote either candidate or launch more training.
Candidates preserve optimizer/RNG state and provenance; they are separate
experiments and do not overwrite the original migration run. On interruption,
the report records interruption/failure and any completed saves remain. This
runner does not implement automatic resume of interrupted experiments.

Run on Colab CUDA using the repository's existing environment:

```bash
python -u runtime/experiments/phi_runtime_lab/coherence_optimizer_trial.py \
  --checkpoint /content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1/run_60k_to_61k/latest.pt \
  --data-dir /content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1/corpus_repaired_v1 \
  --output /content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1/optimizer_trial_NEW_UNIQUE_NAME \
  --device cuda --steps 200
```

Training remains PyTorch/CUDA. Native C++ implements inference, not backward
propagation or optimizer updates. A better result in this short pilot is a
reason to investigate a bounded continuation, not proof that coherence will
emerge or that a broad chatbot is trained. Establish useful text continuation
and evaluate short, consistently formatted instruction examples before claiming
question-answering capability. Do not erase code indentation or retrain the
tokenizer in place.

Local verification: `python test_coherence_optimizer_trial.py` runs a synthetic
end-to-end mini model; checks batch equality, preserved source bytes, resumed
Adafactor versus fresh AdamW optimizer histories, saved step/token counts,
actual parameter updates, byte-whitespace classification and output refusal.
The user's Drive checkpoint is not available in the development environment.
