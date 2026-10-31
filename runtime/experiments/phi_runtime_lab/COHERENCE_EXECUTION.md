# Bounded coherence recovery and native sampling

This workflow compares one correction to the existing optimizer policy. It does not promise that a fixed number of updates produces a general assistant.

Training uses PyTorch/CUDA. Native inference uses the existing RFTMW C++ FP32 CPU model. Sampling controls change token selection, not the model forward implementation.

T4 and L4 can use the FP32 training path; there is no BF16 requirement. Before training, a read-only preflight compares the actual checkpoint's 512-token CPU/GPU forward and backward results, records device memory and numerical differences, and halts on failure. Moving to T4 does not require discarding the saved weights. CPU tests are not evidence of actual T4 execution.

## What changed

- An opt-in parameter-group policy removes weight decay from scalar/vector controls, normalization scales, and biases, including relative-attention bias. It preserves matrix decay and optimizer moments.
- Two independent arms start from the same saved broad SFT checkpoint and sampler state. The default is at most 500 additional updates per arm. Existing runs are not overwritten.
- `best_loss.pt` records loss improvement. `candidate.pt` additionally requires the frozen generated-answer development checks and replay retention. Falling loss cannot by itself qualify a candidate.
- Actual source draws, unique examples and supervised-token exposure are saved. The original data mixture is preserved for this comparison; this does not expand or rebalance the corpus.
- The selected candidate receives one separate final evaluation. A failed final check cannot be reused for selecting another candidate in the same experiment.
- Native export happens only after qualification and C++ logit parity. The fixed checks are a functional smoke suite, not broad assistant certification.
- Native chat supports temperature, top-p, top-k, generated-token repetition penalty and a fixed seed. Default legacy calls remain greedy.

## Run in an existing Colab checkout

```bash
python -u runtime/experiments/phi_runtime_lab/coherence_recovery.py \
  --root /content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1 \
  --device cuda --steps 500
```

Inputs are `broad_continuation_v1/latest.pt`, `corpus_repaired_v1`, and `chat_recovery_v1/data` under that root. Outputs go to the new `coherence_execution_v1` directory. The source checkpoint is copied and hash-verified before training.

Rerunning the command resumes only the fixed budget. Completed experiments do not extend automatically. A changed source, data manifest or budget is rejected. Ctrl+C during an optimizer arm requests a saved stop after its current complete update; a hard runtime deletion may lose work after the last atomic checkpoint. An interrupted final evaluation can rerun deterministically, but a recorded final result is reused only for the identical candidate and suite.

For a persistent stop request, create `coherence_execution_v1/STOP`. Resume that fixed experiment deliberately with `--clear-stop`. `progress.json` in each arm and the top-level `report.json` distinguish stopped, failed quality, failed final checks, and qualified outcomes.

## Native sampled chat after qualification

```python
from pathlib import Path
from chat_recovery_cpp import chat
run = Path('/content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1/coherence_execution_v1/qualified_native')
chat(run, decode_config=dict(temperature=0.7, top_p=0.9, top_k=50,
                             repetition_penalty=1.1, seed=20261031))
```

An existing export compiled before sampling support needs re-export with the new code; its verified binary is not silently replaced. For code, repetition penalties can suppress legitimate repeated identifiers or indentation, so compare with `repetition_penalty=1.0` as well. Reduced repetition is not equivalent to factual or functional correctness.

## Interpretation and contingency

Do not continue this experiment indefinitely if neither arm passes. Preserve the failed results. Separately test update-scale calibration, a broader task-balanced corpus, or a clean-start simple-language comparison against conventional attention. Do not combine all interventions and attribute any difference to weight decay.

The prepared Python set is small and concentrated in recurring task families, and the original source mixture underexposes support and math. Those remain data-recipe limitations; this controlled optimizer comparison intentionally does not conceal them by changing datasets at the same time.

## Local checks

```bash
cd runtime/experiments/phi_runtime_lab
python -m unittest -v test_optimizer_policy test_coherence_quality \
  test_coherence_recovery_train test_coherence_recovery test_native_sampling test_device_preflight
python test_chat_recovery.py
python test_journal_control.py
```

CPU tests validate checkpoint handling, selection/stop behavior, masked targets, native parity and sampling mechanics. Actual L4 training performance and resulting language quality require execution in the Colab runtime.
