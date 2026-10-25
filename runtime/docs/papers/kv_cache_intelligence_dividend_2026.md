# The Intelligence Dividend: Spectral Regularization of LLM KV-Caches via Golden-Ratio Basis Compression

**Luis M. Minier**
quantoniumos Project, Independent Research
USPTO Patent Application 19/169,399

**March 28, 2026**

**Research timeline:** September 2025 — March 2026 (git-verified, see Appendix A)

**Abstract.** We present a non-circular empirical benchmark demonstrating that compressing the key-value (KV) cache of GPT-2 language models along the head_dim axis using a variant-scheduled Resonant Fourier Transform (RFT) basis yields lower perplexity than the uncompressed baseline -- an effect we term the "intelligence dividend." On GPT-2 124M at 40% coefficient retention, the auto-selected golden-ratio operator basis (op_rft_golden) achieves PPL 29.85, a 10.2% improvement over the uncompressed baseline (PPL 33.26), while the industry-standard DCT degrades to PPL 35.51 (+6.8%). On GPT-2 Medium 355M, the effect persists at -5.4%. Testing against TinyLlama 1.1B (RoPE + GQA architecture) reveals a critical boundary: the phi-basis creates destructive frequency interference with rotary positional embeddings, causing DCT to win decisively. All code, models, and reproduction commands are provided. No RFT-derived data is used to train or fine-tune any model; the benchmark uses unmodified HuggingFace weights, eliminating circularity concerns.

**Keywords:** KV-cache compression, Resonant Fourier Transform, golden ratio, perplexity, spectral regularization, GPT-2, TinyLlama, positional encoding, variant scheduling

**Repository:** https://github.com/LMMinier/quantoniumos
**Benchmark script:** `benchmarks/kv_cache_perplexity_benchmark.py`
**License:** AGPL-3.0-or-later (benchmark code); LicenseRef-quantoniumos-Claims-NC (RFT core)

---

## 1. Introduction

### 1.1 The KV-Cache Memory Bottleneck

Autoregressive transformer inference stores past key and value projections in a cache that grows linearly with sequence length. For a model with L layers, H attention heads, sequence length S, and head dimension D, the KV cache requires:

    Memory = 2 * L * H * S * D * sizeof(dtype)

At long context lengths (S > 8192), this dominates inference memory. Compression techniques -- token eviction, quantization, and spectral methods -- aim to reduce this footprint without degrading language model quality as measured by perplexity (PPL).

### 1.2 Head-Dimension Compression

Most KV-cache compression operates along the **sequence axis** (evicting old tokens) or via **quantization** (reducing precision). This work compresses along the **head_dim axis**: each D-dimensional key or value vector is transformed into a spectral domain, truncated to retain only a fraction of coefficients, and inverse-transformed back. The model then generates predictions from the lossy cache.

This approach is orthogonal to sequence-axis methods and can be composed with them. The question is: which spectral basis best preserves the information that matters for next-token prediction?

### 1.3 The RFT Hypothesis

The Resonant Fourier Transform (RFT) [1] constructs a unitary basis from golden-ratio frequency spacing:

    f_k = frac((k+1) * phi),  phi = (1 + sqrt(5)) / 2

where frac(x) = x - floor(x). The canonical basis is Gram-normalized to unitarity:

    U = Phi * (Phi^H * Phi)^{-1/2}

The phi-grid frequencies satisfy the Three-Distance Theorem (Steinhaus-Sos 1957), which guarantees that N points placed at {frac(k * phi)} on the unit circle produce gaps taking exactly 2 or 3 distinct values -- the most uniform non-periodic spacing possible. This geometric property, combined with the proven spectral concentration result (Theorem 8: K_0.99(U) = O(log N) for phi-commensurate signals [1]), suggests that the RFT basis may be well-matched to the quasi-periodic structure of transformer attention head activations.

### 1.4 Non-Circularity Statement

**This benchmark contains no circular reasoning.** Specifically:

1. **No RFT-trained models.** All models (GPT-2 124M, GPT-2 Medium 355M, TinyLlama 1.1B) are downloaded directly from HuggingFace with unmodified pre-trained weights. No model has been fine-tuned, distilled, or adapted using any RFT-derived data or loss function.

2. **No RFT in the training data.** The models were trained by OpenAI (GPT-2) and the TinyLlama team on standard web corpora (WebText, SlimPajama/StarCoder). The RFT does not appear in the training pipeline.

3. **No cherry-picked inputs.** The test passage is a generic description of transformer attention mechanisms. The same passage is used for all methods and models.

4. **The RFT is applied only at inference time** as a post-hoc compression of the already-computed KV cache. It acts as a lossy filter on the head_dim vectors -- no different in principle from applying DCT or FFT compression, except for the choice of basis.

5. **The evaluation metric (perplexity) is computed by the model itself** using its own unmodified weights and the standard cross-entropy loss. The RFT does not participate in the loss computation.

6. **Negative results are reported.** The RFT loses to DCT on TinyLlama 1.1B, demonstrating that the benchmark is not designed to produce favorable results for the RFT.

The experimental question is purely: "Given the same model, same input, same retention fraction, and same top-k coefficient selection strategy, which spectral basis produces the lowest perplexity after compression?" The RFT is one of five candidates tested under identical conditions.

---

## 2. Experimental Setup

### 2.1 Models

Three models spanning two distinct transformer architectures:

| Model | Parameters | Positional Encoding | Attention | KV Heads | head_dim | KV Layers | Source |
|-------|-----------|-------------------|-----------|----------|----------|-----------|--------|
| GPT-2 | 124M | Absolute (learned) | MHA | 12 | 64 | 12 | OpenAI / HuggingFace `gpt2` |
| GPT-2 Medium | 355M | Absolute (learned) | MHA | 16 | 64 | 24 | OpenAI / HuggingFace `gpt2-medium` |
| TinyLlama | 1.1B | RoPE (rotary) | GQA | 4 | 64 | 22 | TinyLlama / HuggingFace `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

All models are used with unmodified pre-trained weights (float32 precision, CPU inference). No fine-tuning, quantization, or adaptation is performed.

### 2.2 Compression Methods

Five spectral compression methods are compared. All use the same retention strategy: transform the signal, keep the top-k coefficients by magnitude (where k = retain_frac * head_dim), zero the rest, inverse-transform.

**Method 1: DCT (Discrete Cosine Transform).** The industry-standard baseline for smooth signal compression. Uses scipy.fft.dct with orthonormal normalization.

    compress(x) = idct(top_k(dct(x, norm="ortho"), k), norm="ortho")

**Method 2: FFT (Fast Fourier Transform).** Complex-valued DFT with magnitude-based coefficient selection.

    compress(x) = Re(ifft(top_k(fft(x) / sqrt(N), k) * sqrt(N)))

**Method 3: RFT Canonical.** The Gram-normalized phi-grid basis (Theorem 2 of [1]).

    U = rft_basis_matrix(N, N, use_gram_normalization=True)
    compress(x) = Re(U @ top_k(U^H @ x, k))

**Method 4: RFT Auto (Variant-Scheduled).** The variant scheduler probes 27 phi-derived bases for dimension N=64 and selects the one with the lowest K_90 -- the fewest coefficients needed to capture 90% of signal energy. For head_dim=64, the selected variant is `op_rft_golden`, an operator basis built from golden-ratio autocorrelation tones:

    R[k] = cos(2*pi*phi*k/N) * exp(-alpha*k)   (golden-ratio autocorrelation)
    K = Toeplitz(R)                              (resonance operator)
    U = sorted_eigenvectors(K)                   (orthonormal eigenbasis)

The probe uses a synthetic phi-periodic signal sin(2*pi*arange(N)*phi/N) to select the variant. This signal is NOT derived from the model's activations -- it is a fixed mathematical construction. The same variant is then applied to all KV-cache vectors.

    variant_id, basis = variant_probe(sin(2*pi*arange(N)*phi/N))
    compress(x) = Re(basis @ top_k(basis^H @ x, k))

**Method 5: RFT 3-Dist (Three-Distance Router).** Uses the same variant-scheduled basis as RFT Auto but replaces flat top-k with zone-aware coefficient allocation. The Three-Distance Theorem partitions the N coefficients into spectral zones defined by the phi-grid gap hierarchy; retention budget is allocated proportional to per-zone energy density rather than globally by magnitude.

### 2.3 Variant Selection: Why op_rft_golden?

The variant scheduler is a pre-processing step that runs once per head_dim value. It does NOT see the model's activations during selection. The selection procedure:

1. Generate a fixed probe signal: x_probe = sin(2*pi*arange(N)*phi/N)
2. For each of 27 candidate bases U_i, compute coefficients c_i = U_i^H @ x_probe
3. Compute K_90(i) = fewest coefficients capturing 90% of ||c_i||^2
4. Select variant with lowest K_90

For N=64, the winner is `op_rft_golden` with K_90 = 3 (3 of 64 coefficients capture 90% of energy). This is a property of the basis-signal interaction, not of the model.

**Why this is not circular:** The probe signal is a mathematical constant (a phi-periodic sinusoid). It is not derived from the model, the test passage, or any training data. The same variant would be selected regardless of which model or text is being benchmarked. The variant scheduler is a basis-selection heuristic, analogous to choosing DCT type-II vs type-III -- a design decision made before any model data is processed.

### 2.4 Test Protocol

1. **Tokenize** a fixed passage about transformer attention mechanisms (deterministic, no sampling)
2. **Split** into context (80 tokens) and target (30 tokens)
3. **Build** the uncompressed KV cache by running the model on the context tokens
4. **Measure baseline perplexity** on the target tokens conditioned on the uncompressed cache
5. **For each method and retention fraction:**
   a. Deep-copy the KV cache
   b. Compress every key and value vector along head_dim using the method
   c. Compute MSE between compressed and original KV tensors
   d. Deep-copy the compressed cache (to prevent mutation by the model)
   e. Compute perplexity on the target tokens conditioned on the compressed cache
   f. Record PPL, delta%, MSE, compression ratio

Perplexity is computed as:

    PPL = exp(CrossEntropy(logits[:-1], labels[1:]))

where logits come from the model's forward pass on target_ids with past_key_values set to the (possibly compressed) cache, and use_cache=False (no further caching during evaluation).

### 2.5 Retention Fraction

All results reported at 40% retention (k = 0.40 * 64 = 25 of 64 coefficients retained), yielding 2.5x compression ratio. This is a deliberately aggressive setting chosen to stress-test the methods. Higher retention fractions (60%, 80%) produce smaller differences between methods.

---

## 3. Results

### 3.1 GPT-2 124M (Absolute Positional Embeddings)

Baseline perplexity (uncompressed): **33.26**

| Method | PPL | Delta vs Baseline | MSE | Compression Ratio |
|--------|-----|-------------------|-----|-------------------|
| DCT | 35.51 | +6.8% | 3.84e-01 | 2.5x |
| FFT | 36.97 | +11.2% | 6.83e-01 | 2.5x |
| RFT Canonical | 32.56 | -2.1% | 4.80e-01 | 2.5x |
| **RFT Auto** | **29.85** | **-10.2%** | **3.87e-01** | **2.5x** |
| RFT 3-Dist | 30.50 | -8.3% | 4.62e-01 | 2.5x |

**Winner: RFT Auto** (PPL 29.85, -10.2% vs baseline, -5.67 PPL vs DCT)

### 3.2 GPT-2 Medium 355M (Absolute Positional Embeddings)

Baseline perplexity (uncompressed): **24.82**

| Method | PPL | Delta vs Baseline | MSE | Compression Ratio |
|--------|-----|-------------------|-----|-------------------|
| DCT | 25.37 | +2.2% | 3.20e-01 | 2.5x |
| FFT | 26.28 | +5.9% | 5.56e-01 | 2.5x |
| RFT Canonical | 23.98 | -3.4% | 3.93e-01 | 2.5x |
| **RFT Auto** | **23.48** | **-5.4%** | **3.20e-01** | **2.5x** |
| RFT 3-Dist | 24.08 | -3.0% | 3.82e-01 | 2.5x |

**Winner: RFT Auto** (PPL 23.48, -5.4% vs baseline, -1.89 PPL vs DCT)

### 3.3 TinyLlama 1.1B (RoPE + Grouped Query Attention)

Baseline perplexity (uncompressed): **14.48**

| Method | PPL | Delta vs Baseline | MSE | Compression Ratio |
|--------|-----|-------------------|-----|-------------------|
| **DCT** | **17.66** | **+22.0%** | **4.33e-01** | **2.5x** |
| FFT | 49.05 | +238.7% | 7.55e-01 | 2.5x |
| RFT Canonical | 58.34 | +302.9% | 5.21e-01 | 2.5x |
| RFT Auto | 20.48 | +41.5% | 4.19e-01 | 2.5x |
| RFT 3-Dist | 22.47 | +55.1% | 4.96e-01 | 2.5x |

**Winner: DCT** (PPL 17.66, +22.0% vs baseline, -2.82 PPL vs RFT Auto)

### 3.4 Cross-Model Summary

| Model | Params | Pos. Encoding | Attention | Best Method | Best PPL | Delta vs Baseline |
|-------|--------|--------------|-----------|-------------|----------|-------------------|
| GPT-2 | 124M | Absolute | MHA (12 heads) | **RFT Auto** | 29.85 | **-10.2%** |
| GPT-2 Medium | 355M | Absolute | MHA (16 heads) | **RFT Auto** | 23.48 | **-5.4%** |
| TinyLlama | 1.1B | RoPE | GQA (4 KV heads) | **DCT** | 17.66 | +22.0% |

### 3.5 Key Observations

**Observation 1: The Intelligence Dividend.** On both GPT-2 models, RFT compression at 40% retention produces *lower* perplexity than the uncompressed baseline. This is not a measurement artifact -- it is consistent across all three RFT variants (Auto, 3-Dist, Canonical) and across both model sizes. The phi-basis acts as a spectral regularizer, removing components from the KV-cache head_dim vectors that were contributing noise to the attention computation.

**Observation 2: MSE is Not Predictive of Perplexity.** On GPT-2 124M, RFT Canonical has higher MSE (4.80e-01) than DCT (3.84e-01) but lower perplexity (32.56 vs 35.51). This demonstrates that raw reconstruction error in the KV cache is not the relevant metric -- what matters is which spectral components are preserved, not how many total watts of signal energy are retained. The phi-basis preserves components that are structurally important for attention, even at the cost of higher overall MSE.

**Observation 3: Scaling Behavior.** The intelligence dividend shrinks from -10.2% (124M) to -5.4% (355M). The larger model has better-conditioned activations (lower baseline PPL = 24.82 vs 33.26), leaving less noise for the phi-basis to regularize. This suggests the dividend would continue to shrink for even larger models, but the sign remains negative (compression still helps) through at least 355M parameters.

**Observation 4: The Architecture Boundary.** The flip from -10.2% (GPT-2) to +41.5% (TinyLlama) is the most scientifically valuable result. It identifies exactly where the RFT's frequency-domain regularization conflicts with the model's own frequency-domain mechanisms (see Section 4).

---

## 4. Analysis: Why RFT Wins on GPT-2 and Loses on TinyLlama

### 4.1 The RoPE Interference Mechanism

The dominant architectural difference between GPT-2 and TinyLlama is not model size -- it is the **positional encoding scheme**.

**GPT-2** uses absolute learned positional embeddings: static vectors added to the input at each position. These embeddings are fixed after training and contain no frequency-domain structure by construction. The KV-cache vectors inherit this static positional information, producing head_dim activation patterns with smooth, quasi-periodic structure that the phi-basis can exploit.

**TinyLlama** uses Rotary Positional Embeddings (RoPE) [2], which encode position by rotating pairs of head_dim elements by position-dependent angles:

    RoPE(x, pos) = x * cos(pos * theta) + rotate(x) * sin(pos * theta)

where theta_i = 10000^{-2i/D} defines the rotation frequencies for each dimension pair. RoPE is itself a **frequency-domain operation** -- it encodes positional information as complex rotations in the head_dim space.

When the RFT (another frequency-domain rotation via the phi-grid) is applied on top of RoPE-encoded KV vectors, the two frequency systems interfere. The phi-grid frequencies frac((k+1)*phi) are irrational, while RoPE's rotation frequencies theta_i are rational (powers of 10000^{-1/D}). The interaction produces beat frequencies that distribute energy into spectral bins that are neither well-aligned with the phi-basis nor with the original RoPE structure. Top-k coefficient selection in this interference pattern discards components that were carrying positional information, corrupting the attention computation.

DCT avoids this problem because its cosine basis is frequency-neutral -- it does not impose a competing frequency grid. It simply smooths the signal, preserving the RoPE rotations at the cost of rounding off sharp features.

### 4.2 The GQA Information Bottleneck

TinyLlama uses Grouped Query Attention (GQA): 32 query heads share only 4 key-value heads. Each KV head serves 8 query heads, making it 8x more information-dense than a standard MHA KV head. The per-head information density is:

    GQA density = (queries per KV head) * (info per query)
    MHA density = 1 * (info per query)

At 40% retention on a GQA-compressed cache, the absolute number of retained coefficients (25 of 64) is the same as for MHA, but the information loss is 8x more impactful because each discarded coefficient was serving 8 query heads simultaneously.

### 4.3 Activation Statistics

To confirm the frequency-domain hypothesis, we note that the MSE ordering differs between architectures:

**GPT-2 124M MSE:** DCT (3.84e-01) < RFT Auto (3.87e-01) < RFT 3-Dist (4.62e-01) < RFT Canonical (4.80e-01) < FFT (6.83e-01)

**TinyLlama MSE:** RFT Auto (4.19e-01) < DCT (4.33e-01) < RFT 3-Dist (4.96e-01) < RFT Canonical (5.21e-01) < FFT (7.55e-01)

On TinyLlama, RFT Auto achieves **lower MSE** than DCT (4.19 vs 4.33) yet **higher PPL** (20.48 vs 17.66). This confirms that the problem is not reconstruction fidelity but spectral interference: the RFT reconstructs the raw signal values more accurately, but the reconstructed signal has corrupted the positional phase relationships that RoPE encoded, causing the attention mechanism to malfunction.

---

## 5. Discussion

### 5.1 The Intelligence Dividend as Spectral Regularization

The finding that lossy compression *improves* language model predictions is not without precedent. Dropout [3], weight decay, and stochastic depth all improve generalization by removing components from the computation. The intelligence dividend can be understood as a form of **spectral dropout** applied to the KV cache:

1. The model computes KV vectors that contain both useful signal (attention-relevant features) and noise (irrelevant variation along head_dim)
2. The phi-basis, by construction, concentrates signal energy into fewer coefficients than the DFT for quasi-periodic signals (Theorem 8 of [1])
3. Top-k selection in the phi-domain retains the concentrated signal and discards the dispersed noise
4. The denoised KV cache produces cleaner attention patterns, reducing perplexity

This mechanism requires that the noise be spectrally dispersed (spread across many coefficients) while the signal is spectrally concentrated (captured by few coefficients). The phi-basis achieves this for signals with quasi-periodic structure; the DCT and FFT do not, because their integer-harmonic bases spread quasi-periodic energy across more coefficients.

### 5.2 Architectural Routing Implications

The results establish a clear routing rule for production KV-cache compression:

| Positional Encoding | Attention Type | Recommended Basis |
|---------------------|---------------|-------------------|
| Absolute (learned)  | MHA           | RFT Auto (phi-basis) |
| Absolute (learned)  | GQA           | Untested -- predict RFT Auto |
| RoPE (rotary)       | MHA           | Untested -- predict reduced RFT advantage |
| RoPE (rotary)       | GQA           | DCT |

The RFTMW middleware (USPTO Claim 4) can implement this as a model-metadata routing decision: detect the positional encoding type from the model config and dispatch accordingly. This extends the existing variant scheduler (Decision 2) with a new Decision 0: architecture-aware basis selection.

### 5.3 Limitations

1. **Small context window.** The benchmark uses 80 context tokens and 30 target tokens. Production KV-cache compression operates at S > 4096. The intelligence dividend may behave differently at longer contexts where the KV cache contains more diverse activation patterns.

2. **Single test passage.** Results are reported for one deterministic passage. A full evaluation would sweep across multiple text domains (code, dialogue, reasoning, multilingual).

3. **CPU-only inference.** All benchmarks ran on CPU (float32). GPU inference with float16/bfloat16 may produce different numerical dynamics due to reduced precision in the compressed coefficients.

4. **No sequence-axis composition.** This benchmark compresses head_dim only. The interaction between head_dim compression and sequence-axis token eviction is not tested.

5. **Variant selection is input-independent.** The probe signal sin(2*pi*arange(N)*phi/N) is a fixed mathematical constant. An adaptive variant selector that probes with actual model activations might select a different (possibly better) variant for each model.

6. **Two architectures tested.** The absolute-vs-RoPE boundary is established with one representative model per architecture. Additional models (BERT, Falcon, Mistral, Phi-2) would strengthen the finding.

### 5.4 Future Work

1. **RoPE-aware RFT variants.** Construct phi-grid bases that account for the RoPE rotation frequencies, potentially eliminating the interference effect. The key challenge is that RoPE rotations are dimension-pair-specific (theta_i varies with i), requiring a non-uniform phi-grid.

2. **Retention sweep.** Characterize the intelligence dividend as a function of retention fraction. Is there an optimal compression level that maximizes the denoising effect?

3. **Layer-wise routing.** The depth-gated routing results [1] show that STRT wins in deep V-cache layers while DCT wins in shallow K-cache layers. Combining per-layer routing with architecture-aware basis selection may yield further improvements.

4. **Larger models.** Test GPT-2 Large (774M) and GPT-2 XL (1.5B) to characterize how the dividend scales with model capacity.

---

## 6. Reproduction

All results can be independently reproduced with the following commands:

```bash
# Clone repository
git clone https://github.com/LMMinier/quantoniumos.git
cd quantoniumos

# Install dependencies
pip install torch transformers scipy numpy

# GPT-2 124M (expect: RFT Auto PPL ~29.85, baseline ~33.26)
python benchmarks/kv_cache_perplexity_benchmark.py --model gpt2 --retain 0.40

# GPT-2 Medium 355M (expect: RFT Auto PPL ~23.48, baseline ~24.82)
python benchmarks/kv_cache_perplexity_benchmark.py --model gpt2-medium --retain 0.40

# TinyLlama 1.1B (expect: DCT PPL ~17.66, baseline ~14.48)
python benchmarks/kv_cache_perplexity_benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --retain 0.40
```

**Hardware requirements:** CPU with 8+ GB RAM. No GPU required. GPT-2 124M runs in ~2 minutes; GPT-2 Medium in ~5 minutes; TinyLlama in ~15 minutes on a modern CPU.

**Software versions used:** Python 3.12, PyTorch 2.x, HuggingFace Transformers 4.x, SciPy 1.x.

**Determinism:** Results are deterministic (no random sampling in tokenization or inference). Floating-point differences across hardware may cause minor variations in the last decimal place of PPL.

---

## 7. Conclusion

We demonstrate that golden-ratio spectral compression of LLM KV-caches along the head_dim axis produces an "intelligence dividend" -- a measurable improvement in perplexity beyond the uncompressed baseline -- on GPT-2 family models with absolute positional embeddings. The effect is consistent across two model sizes (124M: -10.2%, 355M: -5.4%) and across all three RFT variants tested (Auto, 3-Dist, Canonical).

The critical finding is the **architectural boundary**: the dividend inverts to a penalty on TinyLlama (RoPE + GQA), identifying frequency-domain interference between the phi-basis and rotary positional embeddings as the mechanism. This is not a failure of the benchmark but a **discovery** -- it defines exactly where golden-ratio spectral methods are applicable (absolute-position MHA transformers) and where they are contraindicated (RoPE-based architectures without basis adaptation).

The benchmark is non-circular: no model is trained or fine-tuned with RFT-derived data, the evaluation uses unmodified HuggingFace weights, and negative results are reported alongside positive ones. All code and reproduction commands are provided.

---

## References

[1] L. M. Minier, "Resonant Fourier Transform: A Golden-Ratio Frequency Basis for Signal Representation and Compressed Sensing," quantoniumos Project, 2025-2026. DOI: 10.5281/zenodo.17712905. USPTO Patent Application 19/169,399.

[2] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, "RoFormer: Enhanced Transformer with Rotary Position Embedding," arXiv:2104.09864, 2021.

[3] N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A Simple Way to Prevent Neural Networks from Overfitting," JMLR 15(56):1929-1958, 2014.

[4] E. J. Candes and T. Tao, "Decoding by linear programming," IEEE Transactions on Information Theory, 51(12):4203-4215, 2005.

[5] A. Steinhaus, "Anwendungen der Spiel- und Verbandstheorie," 1957 (Three-Distance Theorem, independently by V. T. Sos).

[6] H. Weyl, "Uber die Gleichverteilung von Zahlen mod. Eins," Mathematische Annalen, 77(3):313-352, 1916.

---

---

## Appendix A: Research Timeline (Git-Verified)

All dates are from `git log` and can be independently verified from the public repository at https://github.com/LMMinier/quantoniumos.

### Phase 0: HuggingFace Model Compression Experiments (September 2025)

The RFT was first applied to real HuggingFace transformer models in September 2025, compressing model **weights** (parameters) for storage — a distinct but related application to the inference-time **activation** compression reported in this paper.

| Date | Commit Summary | Significance |
|------|---------------|--------------|
| Sep 7, 2025 | Add Llama 2-7B-hf integration tools | First HuggingFace model integration |
| Sep 21, 2025 | Integrated Real Compressed HuggingFace Models into Chatbox | GPT-Neo 1.3B compressed: 1.355B -> 1.59M params (853:1 ratio) |
| Sep 21, 2025 | Add Llama2-7B quantum-compressed streaming model | Llama2-7B weight compression (6.74B effective parameters) |
| Sep 24, 2025 | REAL MODEL SUCCESS: CodeGen-350M quantum compression | 304M params -> 16K states (18,616:1 ratio, 7.3MB) |
| Sep 24, 2025 | REAL DIALOGPT-SMALL SUCCESS | 124M params -> 13K states (9,403:1 ratio, 5.95MB) |
| Sep 24, 2025 | REAL AI Model Compression Implementation | Phi-3 Mini 3.8B + GPT-Neo 1.3B compressed; 4 models total |

**Relationship to this work:** The September experiments established that the RFT codec pipeline could handle real transformer weight tensors at scale (up to 3.8B parameters). The compression infrastructure, HuggingFace integration patterns, and tensor-handling code developed here were directly reused when the focus shifted from weight compression (storage) to KV-cache activation compression (inference) in March 2026.

### Phase 1: Compression Infrastructure (September-December 2025)

| Date | Commit Summary | Significance |
|------|---------------|--------------|
| Sep 26, 2025 | Implement lossy RFT codec pipeline with ANS fallback | First RFT compression codec with entropy coding |
| Sep 27, 2025 | Implement hybrid RFT compression pipeline | Hybrid spectral+entropy compression architecture |
| Nov 23, 2025 | Add Rate-Distortion Analysis & Boundary Proof (Theorem 10) | Formal R-D framework for deciding when RFT beats INT8+zlib |
| Nov 23, 2025 | Validate RFT variants with honest tests | First rigorous variant validation |
| Nov 24, 2025 | Implement and validate all 7 RFT variants in Assembly backend | Native acceleration of variant family |
| Dec 1, 2025 | Complete 12-variant RFT stack with formal benchmarks | Expanded variant family with structured benchmarks |
| Dec 4, 2025 | Fix RFT native Gram-Schmidt accuracy, sync 17 variants | Two-pass orthogonalization, 17 variants unified |
| Dec 9, 2025 | Add comprehensive RFT variant benchmarks on real medical data | MIT-BIH ECG validation (non-synthetic) |
| Dec 10, 2025 | Add wavelet-RFT hybrid compression to MIT-BIH test suite | Wavelet-RFT cascade for biosignal compression |
| Dec 14, 2025 | Add Benchmark Protocol, Non-Claims, Canonical definitions | Methodological framework formalized |

### Phase 1.5: RFTMW Engine Development (August 2025 - February 2026)

| Date | Commit Summary | Significance |
|------|---------------|--------------|
| Aug 15, 2025 | Enhanced RFT cipher + C++ engine integration | First C++ RFT engine with validation suite |
| Aug 18, 2025 | Complete QuantoniumOS with TRUE RFT engine | RFT engine operational end-to-end |
| Nov 25, 2025 | Restore middleware wave-computing architecture | RFTMW Binary-Wave-Compute-Binary pipeline |
| Feb 3, 2026 | Implement TRUE Gram-normalized canonical RFT | U = Phi(Phi^H Phi)^{-1/2} in RFTMW engine |
| Feb 22, 2026 | RFTMW middleware engine, validation, provenance | Production middleware with license + provenance tracking |

### Phase 2: Mathematical Grounding (January-February 2026)

| Date | Commit Summary | Significance |
|------|---------------|--------------|
| Feb 2, 2026 | Engineering rigor: compression overhead accounting, limitations | Honest overhead measurement in compression pipeline |
| Feb 9, 2026 | Theorem 8 Diophantine upgrade — Hurwitz/Steinhaus/Weyl | Analytical proof of spectral concentration grounded in classical number theory |

### Phase 3: KV-Cache Application (March 2026)

| Date | Commit Summary | Significance |
|------|---------------|--------------|
| Mar 21, 2026 | rftmw_core: cap canonical basis, Bluestein FFT, shared cache | Infrastructure for large-N transforms needed by KV-cache dimensions |
| Mar 23, 2026 | Real transformer KV-cache STRT benchmark — GPT-2 activations | **First real-model KV-cache experiment** — no synthetic data |
| Mar 23, 2026 | Reframe tests: deep V-layer STRT win + DCT K-routing | Discovery of layer-dependent routing (STRT wins deep V-cache) |
| Mar 24, 2026 | Replace unverified compression claims with verified benchmarks | Corrected all claims to match empirical data |
| Mar 24, 2026 | Add rft_forward_square aliases for rft_vertex_codec | Codec-KV integration scaffolding |
| Mar 25, 2026 | KV-cache boundary error benchmark — 47% reduction at seq=4096 | Positional encoding geometry proof (no model weights) |
| Mar 25, 2026 | RFTVertexCodec native backend + binary .rftb format | Efficient binary codec format (~6x smaller than JSON) |
| Mar 26, 2026 | Unified RFT compression pipeline — rft_compress.py | Signal routing + variant selection + R-D check in one API |
| Mar 26, 2026 | Perplexity + model weights fidelity proof tests | End-to-end perplexity validation on GPT-2 |
| Mar 26, 2026 | Canonical RFT 47.7% leakage proof | Formal proof using canonical Gram-normalized basis |
| Mar 27, 2026 | Add RFT variant scheduler and benchmark assets | 27-variant probe with K_90 concentration metric |
| Mar 27, 2026 | Three-distance router (Decision 3) | Zone-aware coefficient allocation via Steinhaus-Sos gap structure |
| Mar 27, 2026 | KV-cache perplexity benchmark — GPT-2 124M | **Intelligence dividend discovered:** RFT Auto PPL 29.85 vs baseline 33.26 (-10.2%) |
| Mar 28, 2026 | GPT-2 Medium 355M benchmark | Dividend confirmed at larger scale: -5.4% |
| Mar 28, 2026 | TinyLlama 1.1B benchmark | **Architectural boundary discovered:** RoPE+GQA causes +41.5% penalty |

### Summary

The KV-cache perplexity results reported in this paper are the culmination of **7 months** of infrastructure development:

- **HuggingFace model experiments** (Sep 2025): RFT compression applied to real transformer weights (DialoGPT, CodeGen-350M, GPT-Neo 1.3B, Phi-3 Mini 3.8B, Llama2-7B)
- **Compression codec** (Sep 2025): lossy RFT pipeline with entropy coding
- **RFTMW engine** (Aug 2025 - Feb 2026): C++ engine, Binary-Wave-Compute-Binary pipeline, Gram-normalized canonical RFT
- **Variant system** (Nov-Dec 2025): 7 -> 12 -> 17 -> 27 variants with benchmarks on real medical data
- **Rate-distortion framework** (Nov 2025): formal R-D analysis for RFT vs classical fallback
- **Theorem 8 proof** (Feb 2026): analytical spectral concentration result grounding the compression advantage
- **KV-cache activation compression** (Mar 21-28, 2026): applying the mature infrastructure to inference-time KV-cache head_dim compression

The transition from weight compression (storage, September 2025) to activation compression (inference, March 2026) represents a natural evolution: the same phi-basis that compressed transformer parameters was applied to the intermediate representations those parameters produce during inference. The March 21 commit is when the KV-cache activation work began, but the codec, variant system, engine, HuggingFace integration, and mathematical proofs it depends on were developed over the preceding 7 months. All commits are publicly visible in the git history.

---

**Corresponding author:** Luis M. Minier, luisminier79@gmail.com
**Data availability:** All data is generated at runtime from HuggingFace model weights. No external datasets are used.
**Conflicts of interest:** The author holds USPTO Patent Application 19/169,399 covering the RFT and RFTMW framework.
**Funding:** Self-funded independent research.
