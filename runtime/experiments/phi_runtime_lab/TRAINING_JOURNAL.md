# PhiState L4 training journal

[Open the notebook in Colab](https://colab.research.google.com/github/Anharmoniclabs/quantoniumoslinux/blob/feat/phistate-native-cpp-trial/runtime/experiments/phi_runtime_lab/PhiState_L4_Training_Journal.ipynb)

The notebook consolidates the current project into one reproducible entry point.
Its code worktree is pinned to `284b8d86f6ce8d296e033b7ac2eb1a2dbda33c13`.
It reads the user's existing `whitespace_repair_v1` Drive directory; it does not
recreate the original datasets or silently replace weights.

Run all mounts Drive, loads the code, audits data/checkpoint metadata, creates
training controls, displays saved results, and writes a timestamped JSON audit.
It does not launch training or enter a blocking interactive chat. Optional
preview/export/chat cells have explicit false defaults.

Start / Resume launches an independent copy of the completed conversation run
as a background subprocess. Default additional budget is 2,000 SFT updates;
the target is fixed at the initial copied SFT step plus that budget. This uses
the EXISTING broad prepared datasets and unchanged training recipe, not a new
corpus and not the tiny/curriculum diagnostic examples. This continuation is
experimental, not a demonstrated fix for coherence. A completed or automatically
stopped run is not extended by repeatedly clicking Start.

The independent fork preserves saved weights, optimizer and RNG state. Source
and copy hashes are checked and the existing run locks are respected. Save-and-
Stop creates the trainer's STOP marker; it saves at a complete update boundary.
Refresh reports both process activity and saved progress/logs. PID start-time
and command-line matching prevent treating a reused PID as this worker.

Audit tables distinguish input tokens, supervised answer tokens, multi-turn
examples, checkpoint exposures, and expected sampling proportions. In particular,
20% replay refers to microbatches, not token exposure. State tensor elements
include persistent buffers and are not mislabeled as parameter count. Counts
across ancestor/descendant checkpoints overlap and must not be added.

The notebook records the user's reported limitations: tiny-task memorization
passed, six trained code functions transfer to tested wordings, but arithmetic
and open conversation remain weak. Native C++ parity is separate from answer
quality. The failed generalization export gate remains in force. A new broad
candidate can be exported manually and separately after reviewing generations;
this does not replace the existing conversation export.

Validation: nbformat 4 schema, all 14 Python code cells parsed, clean outputs,
and unique IDs. The offline mini-model fixture exercised exact checkpoint fork,
source preservation, repeat fork, prepared-token accounting, checkpoint inventory,
and separate STOP handling. Colab authentication, the actual GPU job, and the
user's real checkpoint outputs can only be verified in their runtime.
