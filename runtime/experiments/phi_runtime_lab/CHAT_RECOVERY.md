# Conversation training from the repaired 65K checkpoint

This is an answer-only supervised training stage for the existing PhiState MoE,
using the repaired tokenizer and the saved Adafactor optimizer. The 200-update
AdamW experiment supplied by the user regressed: code other-token BPT rose from
6.43684 to 7.64118 and general other-token BPT from 6.95123 to 7.86123. The
Adafactor control was nearly flat (6.45829 and 6.95047). Neither candidate is
promoted. The original 65K checkpoint remains the parent of this new stage.

## Sources inspected and selected

Source revisions and file SHA-256 hashes are pinned in `chat_recovery_data.py`.
Only the named upstream training files are fetched; official validation/test
files are not downloaded. Source README files, license declarations, revisions,
hashes, filtering counts and exclusions accompany the prepared corpus.

| Source | Use |
| --- | --- |
| OpenAssistant/oasst1 | General conversation, including complete multi-turn chains. English reviewed rank-0 assistant answers; deleted, rejected and incomplete chains excluded. Apache-2.0. |
| xTayyub/High-Quality-Synthetic-Python-Dataset-with-Reasoning-Traces-Chain-of-Thought-for-LLM-Fine-Tuning | Instruction to fenced Python solution. Remove an erroneous leading `python` line; reject malformed schema, invalid syntax and implementation-free stubs. Do not execute downloaded code. Use code solutions, not the separate thought_process field. Apache-2.0. |
| bitext/Bitext-customer-support-llm-chatbot-training-dataset | Limited-weight customer-support reply-writing tasks. Keep placeholders; explicitly frame as drafting a reply, rather than giving the model a store identity. CDLA-Sharing-1.0. |
| openai/gsm8k, main/train | Math explanations plus final answers; remove calculator annotations, retain visible solution steps. MIT. |
| Rapidata/text-to-3d-with-llm-code | Excluded: inspected schema contains render/video preferences, not text/code answer targets. |
| shark803thusz/case_study_LLM_casual_reasoning | Excluded: small evaluation workbook containing ground truth and competing, often incorrect model responses. Not ingested as teaching text. |
| APTO-001/llm-math-reasoning-dataset | Excluded: gated, not fetched. Accessible GSM8K training examples supply math instead. |

The xTayyub pinned file actually contains 1,042 flattened records. Its name/card
does not establish the quality of individual rows. Syntax checks are not tests
of functional correctness. Likewise, source review labels do not establish
that every conversation or math answer is factually correct.

## Format and data integrity

All conversational tasks use the same existing tokens and textual template:

```text
<|general|>
User:
question
Assistant:
answer
```

Additional turns repeat the User/Assistant labels. Prompt and answer are encoded
together, with an explicit check that the prompt tokenization is a stable
prefix. The full text must round-trip exactly. No new vocabulary or token IDs
are introduced. No response is truncated and then falsely labelled complete:
examples longer than 512 input tokens are rejected. EOS is appended only to a
complete answer. All user/history/prefix targets are masked with -100; only the
final assistant answer and its true EOS receive language-model loss. This is
explicitly tested at the logit-gradient level.

Exact duplicate examples are removed. Connected split groups join whole OASST
trees, equivalent Python ASTs, Bitext intents, and identical normalized final
questions across sources. About 8% of groups are held out, with at least one
group per source. The trainer re-encodes all examples and checks group/prompt
disjointness before any update. This is not a universal semantic near-duplicate
detector. Unicode separators within JSON strings are preserved by line-wise
JSONL reading; Python str.splitlines must not be used for these files.

## Training and measurement

The run uses PyTorch CUDA. Native C++ still has no training backward pass.
It restores source weights AND Adafactor state/hyperparameters; it does not
repeat the regressing fresh-AdamW configuration. Default target is 2,000 new
SFT optimizer updates, batch1 with accumulation4, up to context512 for chat.
Each microbatch has an 80% chance of conversation training and 20% chance of a
128-token repaired pretraining replay window (55% code/45% general).

Within conversational examples, source sampling weights are 60% general
conversation, 20% Python, 10% math, 10% customer support, regardless of dataset
row counts. Each source is shuffled and cycled rather than repeatedly choosing
one example independently. Both shuffle state and RNG state are saved.

Every 250 steps, evaluate up to 32 fixed held-out examples per source and show
four greedy prompt responses. The selection score is the same weighted mean
of each source's assistant-token BPT. It is not comparable to pretraining BPT.
Also evaluate the same 4,096-token repaired-corpus validation sample. A rise of
more than 0.5 BPT in that replay mean stops the run and prevents the candidate
from replacing this run's best checkpoint. This is an explicit regression
guard, not a guarantee against forgetting. Six non-improving chat evaluations
after step500 also stop the run. Loss reduction alone does not prove coherence.

`latest.pt` saves every100 steps, at evaluation, and after a graceful stop.
`best.pt` is separate and initially contains the baseline. Checkpoints include
optimizer/RNG/order/cursor state and use atomic replacement plus a readback.
Source bytes are never overwritten. `progress.json`, `metrics.jsonl`, and
`generation_*.json` expose current results. An abrupt machine loss can lose work
since the last save. SIGINT/SIGTERM or a `STOP` file requests saving after the
current optimizer update; do not kill the process if a graceful save is wanted.

## One-cell runner and resuming

Run `run_chat_recovery.sh` in the existing Colab GPU environment after installing
`sentencepiece` and `pyarrow`. No torch upgrade or paid job is requested.
Default paths:

```text
/content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1/
  run_60k_to_61k/latest.pt         original 65K parent
  corpus_repaired_v1/             unchanged tokenizer and replay corpus
  chat_recovery_v1/
    source_cache/                 pinned original downloads and README files
    data/                         immutable prepared train/validation data
    run/latest.pt                 resumable conversation checkpoint
    run/best.pt                   best eligible conversation checkpoint
    run/progress.json
    run/generation_00250.json      and later snapshots
    run/native_latest.json        pointer to verified C++ export
```

Re-running the same cell verifies/reuses data and resumes its own saved run.
It does not restart at65K. Set `PHISTATE_SFT_STEPS` to a larger total to extend
the same stage. Settings, source and prepared-data fingerprints must match.
After a deliberate STOP-file stop, remove that file or invoke the trainer with
`--clear-stop` to intentionally resume. Completed, early-stopped, and
retention-stopped runs export the best eligible checkpoint, which may still be
the baseline if no eligible improvement occurred.

The runner refuses a second concurrent chat pipeline in the same runtime.
It does not interrupt or kill other training processes. Run it after the earlier
optimizer trial has completed. Override `PHISTATE_CHAT_ROOT`,
`PHISTATE_REPAIR_ROOT`, `PHISTATE_DATA_DIR`, or `PHISTATE_SOURCE_CHECKPOINT` only
when intentionally selecting another compatible run.

## Native chat

After training, compile/export the best saved checkpoint and compare C++ versus
PyTorch logits along four native-generated chat trajectories. Require both
full/cached allclose and 100% top-1 agreement before publishing the export.
Use the same repaired tokenizer and chat template during inference:

```bash
python chat_recovery_cpp.py chat --run-dir /path/to/chat_recovery_v1/run
```

Or pass `--prompt 'What is quantum physics?'`. Interactive mode retains recent
turns within its 512-token budget; `/reset` clears history. It executes the C++
binary from a local temporary copy because Drive may prohibit execution.
Interactive inference does not import torch. Export/parity verification does.
No repetition suppression is used to conceal poor generations.

## Verification performed before publishing

`test_chat_recovery.py` exercises complete-answer masks and their loss
gradients, EOS handling, length rejection, grouped splits, malformed Python
rejection, preserved Adafactor step history, bit-exact CPU mini-model resume,
STOP handling, source-byte preservation, native export/generation parity,
Unicode JSONL, prepared-data tamper rejection, and the retention stop/best guard.

The complete downloaded source adapters were also exercised with a separate
test tokenizer. This produced usable train/validation pools from all four
selected sources, including multi-turn conversations, and the resulting rows
were re-encoded and checked for split leakage. Those counts are NOT the user's
corpus counts: final length filtering must run with the actual repaired Drive
tokenizer in Colab. The user's 65K checkpoint and conversational training have
not been run in the development environment. Successful tests establish the
pipeline behavior, not successful learning or a competent chatbot.
