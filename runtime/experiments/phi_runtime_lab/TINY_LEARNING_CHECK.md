# Bounded learning check

The completed conversation run reduced held-out answer BPT from 6.18734 to
5.18160, but its generations remained incoherent. This test asks a narrower
question: can the existing model and preserved optimizer learn 16 short,
distinct answers and distinguish the associated questions?

`tiny_learning_check.py` starts from `chat_recovery_v1/run/latest.pt`, preserves
its Adafactor state, and uses the repaired tokenizer and existing chat template.
It trains only a separate diagnostic copy, four examples per update, for at most
800 updates. No data downloads, optimizer replacement, or production checkpoint
promotion occur. This is PyTorch training; the earlier native C++ parity evidence
remains separate. Diagnostic checkpoint format is deliberately not a production
checkpoint format.

Before training it checks answer-only target alignment, whitespace round trips,
full-forward versus cached-step logits, and causality under suffix perturbation.
It checks a nonzero finite embedding update at the first update. Every 100 steps
it scores all training answers and generates all 16 responses greedily. An exact
answer must match the reference text and terminate with EOS. The same answer is
also scored under a paired wrong question: a positive BPT difference favors the
correct question. A prompt win requires a margin above 0.05 BPT.

The test stops early when all 16 training answers match exactly and at least 14
favor their correct prompts. These are diagnostic criteria, not calibrated
capability thresholds. It also prints four unseen question wordings at the end;
these are never used for training or checkpoint selection. Their output needs
semantic review, because multiple phrasings can be correct.

`latest.pt`, `best_diagnostic.pt`, `progress.json`, per-evaluation JSON, and
`report.json` are saved in the independent run directory. The best diagnostic
copy is selected by exact answers, then training BPT. Rerunning with the same
source/settings resumes saved updates. A `STOP` file or SIGINT/SIGTERM requests
saving after the current complete update; remove STOP to resume. A hard process
kill may lose updates since the last evaluation save. A changed source is
rejected. Source and output run locks prevent concurrent training against the
same paths when the other worker uses the chat runner's locking protocol.

Failure within 800 steps does not prove an architecture bug: optimization speed
and the chosen budget are still possible explanations. Success establishes tiny
set memorization and measured prompt sensitivity, not generalization, factual
knowledge, or useful conversational ability. Do not promote this overfit model.

Offline validation: `test_tiny_learning_check.py` exercises a synthetic mini
model, target checks, forward/cache and causality checks, actual weight updates,
bit-exact interrupted/resumed weights, STOP, original checkpoint preservation,
and rejection of a changed source. Tests do not use the user's actual weights
or establish its tiny-task outcome.
