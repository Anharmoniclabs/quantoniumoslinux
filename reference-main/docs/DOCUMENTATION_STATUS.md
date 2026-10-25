# Documentation Status and Triage

**Last reviewed:** 2026-04-16

This file is the repo-wide documentation gate. Use it to decide which documents are safe to cite for the current repository snapshot.

---

## 1. Authoritative for the current snapshot

These are the strongest sources for the actual math, claim boundaries, and inference stack:

- `docs/research/CURRENT_RESEARCH_POSITION.md`
- `docs/THEOREMS_RFT_IRONCLAD.md`
- `docs/PROOF_RESULTS.md`
- `docs/NON_CLAIMS.md`
- `docs/research/OPEN_PROBLEMS.md`
- `docs/research/PROVEN_EMPIRICAL_OPEN_MATRIX.md`
- `docs/QUANTONIUMOS_INFERENCE_ARCHITECTURE.md`
- `docs/RFTMW_INFERENCE_ENGINE_MANUAL.md`

Use these first when describing what is **proven**, what is **empirical**, and what is explicitly a **non-claim**.

---

## 2. Current high-confidence research position

### Proven / strongest

- Canonical discrete RFT definition via Löwdin/polar orthogonalization
- Full rank of the raw φ-grid basis in the discrete setting
- Unitarity and exact reconstruction of the canonical transform
- Distinction from plain DFT rephasing/permutation in the discrete construction
- Exact φ-sparsity result versus dense DFT representation for the targeted signal family

### Empirical / open

- Gram conditioning beyond the resolved Fibonacci cases
- Continuous-time parent and aliasing theory outside the resolved discrete regime
- KV-cache intelligence-dividend results
- RoPE-gap closure / architecture-aware variants
- Medical and biosignal compression wins

### Explicit non-claims / caution zones

- No universal FFT replacement claim
- No proven wave-computer theorem claim
- No standard cryptographic hardness claim for RFT-SIS without new assumptions
- No silicon-level production efficiency claim for the hardware program

---

## 3. Quarantined or rewritten docs

The following docs previously carried stale repo-topology references or unverifiable asset lists. They have been rewritten or quarantined:

- `docs/research/FINAL_AI_MODEL_INVENTORY.md`
- `docs/AI_INFRASTRUCTURE_WALKTHROUGH.md`
- `docs/research/TECHNICAL_SUMMARY.md`
- `docs/reports/RFT_TESTING_SUMMARY.md`
- `docs/project/PROJECT_STRUCTURE.md`
- `docs/project/REPO_ORGANIZATION.md`
- `docs/project/COMPONENT_INVENTORY.md`
- `docs/TECHNICAL_DETAILS.md`

Do **not** use older copies of those documents as proof of current file layout.

---

## 4. Verified repo map

The current repository centers on these top-level areas:

- `algorithms/` — RFT and related algorithm work
- `benchmarks/` — benchmark drivers and experiments
- `docs/` — proofs, manuals, reports, and research notes
- `experiments/` — exploratory studies
- `hardware/` — RTL / FPGA / synthesis artifacts
- `mobile/` — mobile app workspace
- `quantonium_os_src/` — desktop apps, frontend, and engine code
- `src/` — native runtime and application-side utilities
- `tests/` — unit, integration, proof, medical, native, and validation suites
- `tools/` and `ui/` — support scripts and shared assets

---

## 5. Repo hygiene rule

When adding or editing documentation:

1. verify the path exists in this snapshot,
2. separate **proven**, **empirical**, and **open** claims,
3. avoid citing removed directories or old experimental outputs as if they are current.
