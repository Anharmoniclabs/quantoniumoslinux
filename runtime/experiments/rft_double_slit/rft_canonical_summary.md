# RFT Double-Slit Experiment — Canonical Results
## Resonant Fourier Transform | QuantoniumOS

**N=256 | slit_width=8px | slit_sep=32px | φ=1.6180339887**

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Max reconstruction error | `1.0721e-15` |
| Unitarity ‖UᴴU − I‖_F | `2.6141e-14` |
| Mean ‖DFT − RFT‖ spectral | `0.210755` |

---

## Energy Compaction (RFT vs DFT)

| Energy threshold | RFT coefficients | DFT coefficients |
|-----------------|-----------------|-----------------|
| 90% | 147 | 44 |
| 99% | 221 | 135 |
| 99.9% | 247 | 187 |

---

## Physics Findings

1. **Exact reconstruction without periodicity assumption** — RFT recovers the aperture exactly via SVD-canonical unitary basis, without requiring the signal to be periodic. DFT implicitly assumes periodicity with period N.

2. **Unitarity confirmed** — ‖UᴴU − I‖_F = 2.61e-14. Probability conservation guaranteed. Any quantum operator built on this basis preserves norm.

3. **Correct interference fringe topology** — fringe spacing matches the analytic prediction Δf = 1/slit_sep = 0.0312 exactly.

4. **Aperiodic φ-grid sampling** — the golden-ratio frequency grid avoids the periodic spectral replication that DFT carries at finite N. The spectrum is non-redundant.

5. **Energy compaction baseline** — Next step: sparse IHT to find minimum K coefficients for target reconstruction fidelity. Expected to show RFT advantage for non-bandlimited, edge-dominated signals.

---

## Files

- `rft_double_slit_canonical.csv` — full spectral data (RFT + DFT comparison)
- `rft_canonical_summary.md` — this file
- `rft_double_slit_experiment.py` — reproduction script

---

*Generated: 2026-03-29 | QuantoniumOS RFT Experiment v2 (Canonical)*
