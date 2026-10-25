# Production Track (B): Hybrid-as-Production + Bounded Error

This document formalizes the repo's "Track B" position:

- **Canonical RFT** is the mathematical reference operator (slow, verification-grade).
- **Hybrid FFT/RFT** is the production operator (fast, scalable), used when the system needs to run on commodity hardware and avoid datacenter-scale inference.

The goal is not to prove the operators are identical. The goal is to provide **auditable, bounded-error behavior** for the workloads that RFTMW targets (compression + memory bandwidth reduction).

## Operators (as shipped)

### Canonical (reference)

- Source: `algorithms/rft/core/resonant_fourier_transform.py`
- Construction: φ-grid exponential basis + Gram/Löwdin normalization (unitary at finite N).
- Complexity:
  - Basis build: expensive
  - Apply per block: `O(N^2)`

### Hybrid (production)

- Spec: `docs/HYBRID_FFT_RFT_ALGORITHM.md`
- Operator:
  - `Y = E ⊙ FFT(x) / sqrt(N)`
  - `E[k] = exp(i * 2π * frac((k+1) * φ))`
- Complexity:
  - Apply per block: `O(N log N)`
- Implementation status:
  - Used by RFTMW compression blocks when `QUANTONIUM_RFTMW_RFT_IMPL=hybrid`.
  - Native acceleration (if available) is optional; NumPy FFT fallback is acceptable for correctness.

## RFTMW contract (what we claim and test)

RFTMW’s memory layer (`quantonium_os_src/engine/rftmw_memory.py`) implements a **router** and a **guardrail**:

1. **Router**: choose RFT (wave-space) only for structured tensors (low spectral entropy).
2. **Guardrail**: if an RFT attempt exceeds `max_rft_error` (default 0.08 relative error), automatically fall back to INT8+zlib.

This makes the system non-blackbox in a practical, user-facing way: every tensor has a method, an entropy score, and a measured reconstruction error.

## Evidence artifacts

### 1) Canonical-vs-Hybrid "bridge" report (difference quantification)

Run:

```bash
python scripts/bridge_canonical_vs_hybrid.py
```

Writes a JSON report under `results/` that measures:
- unitarity of both operators
- operator-level difference norms
- action-level difference on representative signals

### 2) RFTMW scaling report (production evidence)

Run:

```bash
python benchmarks/rftmw_memory_scaling.py --impl canonical
python benchmarks/rftmw_memory_scaling.py --impl hybrid
```

Writes JSON reports under `results/` with:
- ingest (compression) seconds
- per-tensor decompress latency
- compression ratio
- per-tensor reconstruction error

### 3) Error-sweep report (bounded-error tuning)

Run:

```bash
python benchmarks/rftmw_error_sweep.py --impl hybrid
```

This produces a keep-ratio vs error curve for structured and random signals, to justify default settings and provide tuning guidance.

## “Done” criteria for Track B

Track B is considered “closed” for the production stack when:

- Hybrid mode is the default for production deployments that need scalability.
- Every compressed artifact carries explicit provenance (impl used + params).
- The router+guardrail ensures bounded reconstruction error by construction.
- Reproducible scripts generate evidence reports in `results/` on a fresh machine.

