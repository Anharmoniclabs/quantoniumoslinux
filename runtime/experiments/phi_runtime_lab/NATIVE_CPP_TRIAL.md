# PhiState MoE native C++ inference trial

This adapter runs the complete PhiState MoE token forward pass in C++ on the
CPU, reusing `rftmw::inference::dense_sgemv_f32` from the existing RFTMW engine.
The executable has no PyTorch/LibTorch dependency. Python loads the checkpoint,
exports FP32 weights, tokenizes/decodes text, and supplies a PyTorch CPU reference.

It supports the current DFT model, phi and local modes, top-k routed SwiGLU
experts, an optional shared expert, sliding-window attention, complex recurrent
memory, RMS normalization, tied embeddings, and the channel mixer. The export
folds the checkpoint's fixed spectral mixer and recurrence coefficients into
equivalent inference constants. It does not retrain or quantize weights.

## Run in the Colab terminal after the existing 65K run finishes

```bash
bash runtime/experiments/phi_runtime_lab/run_native_cpp_trial.sh
```

The wrapper expects the existing Drive paths under
`/content/drive/MyDrive/Quantonium/PhiState-L4/whitespace_repair_v1`.
Override `PHISTATE_REPAIR_ROOT`, `PHISTATE_RUN_DIR`, `PHISTATE_DATA_DIR`, or
`PHISTATE_OUTPUT_DIR` if needed. Each trial uses a new output folder.
For other checkpoints invoke `native_cpp_trial.py --help` and supply explicit paths.

The runner verifies checkpoint/corpus identity and repaired tokenizer hash,
compiles locally with `g++ -O3 -march=native -std=c++17 -pthread`, then compares
every logit on 128 validation tokens per domain and four prompts against both
PyTorch full-sequence and cached-step execution. It requires `atol=1e-3`,
`rtol=1e-4` and 100% top-token agreement before native greedy generation.
It checks the generated trajectories too. Validation hashes are checked; the
test split is not used. Exact or near expert-score ties can cause differing
top-k routes and must be investigated if parity fails.

On success the output includes `model.phimoe`, `phistate_moe_cli`, the matching
`tokenizer.model`, and `report.json`. The executable is compiled for the current
CPU; rebuild it on a different machine. Reports include native forward time and
PyTorch CPU cached-step time with the same thread setting. Those are diagnostic
timings, not a controlled speedup benchmark or a CUDA comparison.

The `.phimoe` format is a versioned little-endian FP32 inference export, not the
original resumable `.pt` checkpoint. Original checkpoint/data files are read only.
The C++ CLI consumes raw uint32 token IDs and writes raw FP32 per-position logits:

```bash
phistate_moe_cli model.phimoe tokens.u32 logits.f32 64 EOS_ID
```

## What this establishes

Local verification covered six briefly trained mini configurations across
DFT/phi/local modes, top-1/top-2/top-4 routing and optional shared experts,
including cache eviction, nonidentity mixers and complex readouts. Every tested
position had identical top-token predictions; maximum cached-step logit error
was below `1e-6`. A random model matching the reported 38,287,392-parameter
configuration passed at 128 tokens with maximum error `3.46e-6` and 100% top-token
agreement. Truncated models and out-of-range tokens were rejected.

These local checks did not use the user's trained Drive checkpoint; that check
runs in Colab. No output-quality improvement follows merely from changing the
backend. Native inference should reproduce the trained model's behavior.

## Native training remains separate

This version has no C++ backward pass, gradient accumulation, router-loss
backpropagation, Adafactor optimizer, or native training-state resume. Training
continues to use the existing PyTorch runner. The next native-training milestone
is gradient and single-update parity against PyTorch, including optimizer state,
before any long continuation. A LibTorch C++ frontend would still be a PyTorch
backend and is not what this trial implements.

Run local adapter checks with:

```bash
python runtime/experiments/phi_runtime_lab/test_native_cpp_trial.py
```
