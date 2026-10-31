# Bounded generalization curriculum

The user's tiny-set diagnostic passed at update 400: 16/16 exact training
answers and 16/16 correct-prompt preferences. Two of four unseen wordings were
correct. This motivates testing wording and input transfer; it does not establish
broad capability or rule out every implementation issue.

This run starts from the completed conversational `latest.pt`, never the
memorized diagnostic checkpoint. It preserves Adafactor state and the repaired
tokenizer, uses the existing newline chat template, and trains in PyTorch CUDA.
Outputs are independent under `generalization_v1`; the existing chat remains
unchanged. Default budget is 1,200 updates, four microbatches each.

Training mix:
- 60% reviewed curriculum, equally sampled between knowledge, code, arithmetic.
- 20% previously prepared conversational training data with its existing source weights.
- 20% original repaired corpus, context 128, code share 55%.

The curriculum contains 279 training prompts: 12 short definitions with three
wordings each, six arithmetic Python functions with three wordings each, and
addition/subtraction/multiplication over operands 1..5 with three templates.
Definitions have deliberately short canonical answers. This is a narrow
curriculum, not a replacement for broad language pretraining.

Development and final sets each contain 42 prompts: 12 knowledge rewordings,
six code rewordings, 12 math rewordings with training-range operands, and 12
math problems with new operands. Development uses new operands 6..9; the final
set uses 10..13. The final set is opened only after checkpoint selection.
Questions are exactly disjoint across splits; concepts and code operations are
intentionally shared. Exact development/final question strings are filtered
from conversational replay. Prior pretraining or semantically similar replay
exposure cannot be excluded. These are not unseen-concept evaluations.

Scoring requires EOS. Knowledge scoring conservatively matches the canonical
answer after case/punctuation/whitespace normalization: a differently worded
correct answer may be marked wrong, so outputs are included for review. Math
requires the correct integer alone. Code scoring accepts a single appropriately
named function with the required arguments and a single arithmetic return
expression. A small AST interpreter checks several input cases, including
negative values and zero; it never execs generated Python or permits calls,
imports, attributes or loops. Correct implementations outside this restricted
subset may be marked wrong. Passing finite cases is not a proof for every input.

Checkpoint selection uses the mean of the four development bucket pass rates,
not training loss. The original replay validation mean may worsen at most
0.5 BPT relative to the run's source. A larger increase stops training and
prevents candidate selection. Four development checks without improvement after
at least 600 updates stop the run early. A baseline best.pt is saved; if no
candidate improves, final evaluation transparently uses that baseline.

Native export is optional and isolated to this run. The operational gate is a
development-score gain of at least 0.10, final macro score at least 0.50, and
final knowledge/code scores each at least 0.50. It is not a claim of general
chatbot quality. Eligible candidates use the existing full-trajectory C++ parity
export. Original chat files and pointers are never replaced. Final outputs also
include three open questions for qualitative review.

Checkpoint saving occurs every 100 updates and at evaluations/stops. Rerun the
same cell to resume. STOP or SIGINT/SIGTERM saves after a complete optimizer
update; remove a manually created STOP before resuming. A hard kill can lose
updates since the last save. Source/configuration mismatches fail closed.
Completed runs reuse their final report rather than extending after opening
final questions. A fresh experiment is needed for a changed budget after that.

Files: `latest.pt`, `best.pt`, `progress.json`, `curriculum.json`, development
JSON reports, and `report.json`. `best.pt` is a candidate in this run only.
The default final cell includes --export-native; export is withheld unless the
gate passes, and failures can be retried with the same cell without retraining.

Offline validation: test_generalization.py checks split disjointness, all target
scores, rejection of incorrect/unsupported code, and a mini-model runner with
actual weight updates, exact interrupted/resumed weights, STOP, unchanged source
checkpoint and final-report reuse. It does not run the user's actual model or
claim that this curriculum will cure its coherence problems.
