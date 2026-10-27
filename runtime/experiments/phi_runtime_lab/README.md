# PhiState: a QuantoniumOS-compatible experimental hybrid transformer

**Status:** Original research prototype created for CPU-based experiments. **Not** a pretrained general intelligence model, a replacement for Qwen, or a demonstrated compute breakthrough. This runs **inside** your QuantoniumOS source tree but currently uses PyTorch CPU; it is **not** yet plugged into `rftmwd`, the native GGUF runner, or your native RFTMW/ASM kernels. The C++ port is a separate measured engineering step.

## What this actually implements

- **Local sliding-window causal attention** computed in `O(sequence_length * window * width)` rather than materializing a full `T*T` attention matrix.
- **Phi-frequency diagonal complex state memory:** `s[t] = a * s[t-1] + input[t]`, with golden-ratio frequency initialization and learned decay/readout. Its *training* equivalent is causal real FFT convolution; *inference* processes one byte/token at a time with fixed-size recurrent state. This is inspired by diagonal SSMs/S4D. The phi-frequency choice is experimental and is **not** the canonical RFT itself.
- **True canonical RFT channel mixer** using `U = Phi (Phi^H Phi)^(-1/2)` for blocks of 16 hidden channels. It learns bounded per-frequency gains. The same model can instead use a unitary **DFT** channel mixer for an equal-parameter control. This applies the exact Gram-normalized definition on each channel group.
- **Small SwiGLU feed-forward** and **tied embedding/output** to keep parameters and optimizer memory low.
- **Optional spectral-gradient experiment** (`--grad-filter phi|dft`) projecting *already computed* weight gradients into an orthogonal spectral basis, retaining the top-k coefficients and reconstructing. It tests a sample-efficiency hypothesis; it does *not* save backprop computation, and it may hurt performance.
- **Byte-level next-token learning** (UTF-8 byte IDs 0–255, cross entropy): no costly tokenizer training and easy reproducible holdout BPB. It is not comparable to word/token perplexity without conversion.

All three `phi`, `dft`, and `local` modes use the same local-attention and SwiGLU implementation. `phi` vs `dft` have identical trainable parameter counts; they differ in (1) recurrence frequency initialization and (2) channel mixing basis. `local` is a simpler, smaller ablation without spectral memory/mixer, **not** an equal-parameter control. The `factorial` command **does** isolate phi-memory from channel mixing using a 2×2 ablation (four equal-parameter models).

## Install in your existing Linux/WSL QuantoniumOS repository

Requirements: Python 3.10+, PyTorch CPU, NumPy, git, unzip, working local disk space. Tested here on **PyTorch 2.10 CPU**. On your i5-6600, `--threads 4` is a reasonable starting point; measure `--threads 2` too. No GPU, cloud API, teacher model, or pretrained checkpoint is required for these experiments.

1. Download `phi_runtime_lab.zip` from the ChatGPT response to your computer.
2. In WSL, copy the ZIP to `~/Downloads/phi_runtime_lab.zip` (or adjust the file path in the command).
3. From your QuantoniumOS checkout:

```bash
# If not already cloned:
git clone https://github.com/Anharmoniclabs/quantoniumoslinux.git
cd quantoniumoslinux

unzip ~/Downloads/phi_runtime_lab.zip -d runtime/experiments/
python3 -m venv .phistate-env
source .phistate-env/bin/activate
python -m pip install -U pip
# Free CPU-only wheel; installation requires internet on first use.
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy

bash runtime/experiments/phi_runtime_lab/run_in_repo.sh smoke
bash runtime/experiments/phi_runtime_lab/run_in_repo.sh quick runtime/docs
```

A more informative experiment after the smoke test:

```bash
bash runtime/experiments/phi_runtime_lab/run_in_repo.sh research runtime/docs

# Stronger causal attribution: 2x2 RFT-vs-DFT spectral state and mixer
bash runtime/experiments/phi_runtime_lab/run_in_repo.sh factorial runtime/docs
```

The research profile compares 3 variants on your text, each trained **from scratch** with the same initialization seed, parameter budget (`phi` and `dft`), 500 optimizer steps, and document-level validation/test splitting where at least 10 documents are present. It logs held-out validation **bits per byte** (lower is better), update throughput, parameters, and peak process memory. It takes whatever compute your system actually requires; no speedup is presumed.

## Train one model, evaluate, and generate

If the PDF/manuals aren't usable training data, provide your own **UTF-8 `.txt` file or directory of `.md`, `.txt`, `.py`, `.rs`, `.cpp`, etc.** The trainer excludes `private/`, `secrets/`, `.git/`, `data/`, `models/`, and build folders in recursive directory mode. Review the chosen directory; public repository content can still have personal/sensitive text. The trainer does not send it to a server.

```bash
LAB=runtime/experiments/phi_runtime_lab
python "$LAB/lab.py" train --data runtime/docs --mode phi \
  --width 96 --layers 2 --heads 4 --window 16 --mix-group 16 \
  --context 128 --batch 8 --steps 500 --threads 4 \
  --out "$LAB/checkpoints/my_phi.pt"

# Hold out TEST until architecture/hyperparameters have been selected on validation.
python "$LAB/lab.py" evaluate --checkpoint "$LAB/checkpoints/my_phi.pt" \
  --data runtime/docs --split test --batch 8 --threads 4

python "$LAB/lab.py" generate --checkpoint "$LAB/checkpoints/my_phi.pt" \
  --prompt 'The CPU scheduler' --new-bytes 160 --threads 4
```

A tiny model with a few MB of documentation is a **capability and training experiment**, not a general conversational assistant. Initial output will be noisy until the model is trained substantially; a byte LM can emit invalid UTF-8. A useful assistant would require orders of magnitude more diverse high-quality data, model capacity, optimization and/or distillation from an existing teacher.

## Test *training algorithms*, not just model structures

```bash
bash runtime/experiments/phi_runtime_lab/run_in_repo.sh gradients runtime/docs
```

This runs the same **PhiState** model three times: `none` (ordinary AdamW), `phi` (canonical-RFT top-k gradients), and `dft` (DFT top-k gradients), with equal steps, 50% coefficient retention, and equal starting weights. Compare validation BPB *and* measured training bytes/sec. Filtering every gradient after backprop is extra work; only a demonstrated sample-efficiency gain would motivate a later custom sparse trainer.

## Research protocols and limitations

1. **Before interpreting results**, run `smoke`: it checks phi/DFT unitarity, identity at gain=1, causal future independence, FFT-convolution vs recurrent evaluation, exact full-model vs streaming logits, and valid backward gradients. In the author's development environment, all checks passed.
2. **Always test against controls.** Same corpus, random seed, context, parameter budget, gradient budget; compare both training speed and held-out BPB. Run multiple seeds before describing an improvement as real.
3. **Use document-level splits** for 10+ input text files; smaller corpora use a contiguous 80/10/10 split. Exact duplicate files are removed, but *near*-duplicates may remain. Do not tune against the held-out test split. On a single repeating synthetic file, contiguous splitting is not a meaningful generalization check.
4. **Separate axes** before any performance claims: (a) golden recurrent frequencies vs DFT frequencies, (b) canonical mixer phi vs DFT, (c) optional gradient filter phi vs DFT vs none. The quick `compare` combines (a) and (b) in `--mode`; run `factorial` to isolate them using `--mix-basis phi|dft` with both memory modes. No inference mechanism is presumed superior.
5. **These changes do not eliminate the cost of learning.** Small parameter count, fixed-window attention, and FFT-based training reduce resource requirements, but quality depends on data and compute. RFT unitary structure does not establish a general-purpose sample-efficiency advantage.
6. **Native RFTMW integration is future work:** after controlled superiority, port `GoldenState.step` to `runtime/src/rftmw_native/` as two-float complex SIMD recurrence; port `CanonicalChannelMixer` as a 16×16 cached real matrix; quantize linear weights; compare native versus PyTorch CPU using identical checkpoints. `rftmwd` can be used to launch that compiled executable as a governed workload, but does not train the model itself.

## Key files

- `phistate.py`: the new neural architecture, two modes of spectral processing, and streaming state.
- `lab.py`: reproducible corpus ingestion, training, evaluation, numerical smoke tests, demo data, checkpointing, and A/B comparisons.
- `run_in_repo.sh`: shortcuts for `smoke`, `quick`, `research`, `factorial`, `gradients`.
- `demo.txt`: **synthetic** data for checking that training runs, not a benchmark of intelligence or generalization.

## Research lineage

- S4D, *On the Parameterization and Initialization of Diagonal State Space Models*, Gu et al. (2022): https://arxiv.org/abs/2206.11893
- Mamba, *Linear-Time Sequence Modeling with Selective State Spaces*, Gu & Dao (2023–2024): https://arxiv.org/abs/2312.00752
- Your canonical phi-grid RFT (QuantoniumOS repository) provides the **specific fixed channel basis** tested here. The combined architecture is a testable experiment, not a claim of being globally unprecedented.
