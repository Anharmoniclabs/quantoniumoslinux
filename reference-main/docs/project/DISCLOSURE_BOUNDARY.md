# Disclosure Boundary

**Date:** April 2026
**Purpose:** Define what can be shown publicly, what stays private, and what needs review before sharing.

---

## Three Tiers

### Tier 1 — Public Mirror Safe

Files in this tier can be placed in a public repository or shared with reviewers without restriction.

| Category | Examples |
|----------|----------|
| Proved theorems | `docs/THEOREMS_RFT_IRONCLAD.md` |
| Research position statements | `docs/research/CURRENT_RESEARCH_POSITION.md` |
| Explicit non-claims | `docs/NON_CLAIMS.md` |
| Limitations / concerns | `docs/LIMITATIONS_AND_REVIEWER_CONCERNS.md` |
| Open problems | `docs/research/OPEN_PROBLEMS.md` |
| Classification matrix | `docs/research/PROVEN_EMPIRICAL_OPEN_MATRIX.md` |
| License files | `LICENSE`, `LICENSE-CLAIMS-NC.md`, `LICENSE-SPX-PROPRIETARY.md` |
| README (after PII cleanup) | `README.md` |
| Setup / contribution guides | `SETUP.md`, `CONTRIBUTING.md` |
| Benchmark scripts (not results) | `benchmarks/*.py` (scripts only) |
| Example usage | `examples/` |

**Rule:** Nothing in Tier 1 should contain personal email addresses, internal memos, or unqualified performance claims.

### Tier 2 — Private, Shareable on Request

Files in this tier stay in the private repo but can be shared with specific reviewers, collaborators, or patent counsel on a case-by-case basis.

| Category | Examples |
|----------|----------|
| Core RFT implementations | `algorithms/rft/core/` |
| Crypto implementations | `algorithms/rft/crypto/` |
| Compression / codec pipeline | `algorithms/rft/compression/`, `algorithms/rft/hybrids/` |
| Hardware RTL | `hardware/rtl/`, `hardware/*.sv` |
| Native C++/SIMD kernels | `src/rftmw_native/` |
| ASM implementations | `algorithms/rft/kernels/engines/` |
| Internal benchmark results | `benchmarks/results/`, `data/quantum_compression_results.json` |
| Claims practicing registry | `docs/project/CLAIMS_PRACTICING_FILES.txt` |
| Patent documentation | `docs/patent/` |
| Post-review status | `docs/research/STATUS_AFTER_BINDEL_REVIEW.md` |
| Four-claim map | `docs/research/FOUR_CLAIM_REPO_MAP.md` |

**Rule:** Sharing requires explicit decision. Consider whether the recipient needs implementation details or just the mathematical framework.

### Tier 3 — Private, Do Not Share

Files in this tier contain PII, internal strategy, or frank assessments that should not leave the private repo.

| Category | Examples |
|----------|----------|
| PII index | `docs/private/PII_AND_PATENT_ADMIN_INDEX.md` |
| Internal gap memo | `docs/private/INTERNAL_RESEARCH_GAP_MEMO.md` |
| Personal contact info | Email addresses in `CITATION.cff`, `docs/project/Author` |
| Patent prosecution details | Filing numbers, confirmation numbers, examiner correspondence |
| CI / deployment credentials | `.github/` secrets, tokens |

**Rule:** Never share. If a reviewer needs patent info, point them to the public USPTO record.

---

## Decision Checklist

Before making any file public or sharing it:

1. **Does it contain PII?** (email, phone, home address) → Tier 3 or redact first.
2. **Does it contain unqualified claims?** ("proves quantum advantage," "exponential speedup") → Fix first, then reconsider.
3. **Does it contain implementation details of patent claims?** → Tier 2 minimum.
4. **Is it a proved theorem or honest assessment?** → Tier 1 candidate.
5. **Has the overclaim language been corrected?** → Check against `docs/research/STATUS_AFTER_BINDEL_REVIEW.md`.

---

## Cross-References

- Public mirror plan: see repo migration documentation
- PII inventory: `docs/private/PII_AND_PATENT_ADMIN_INDEX.md`
- Overclaim audit: `docs/research/STATUS_AFTER_BINDEL_REVIEW.md`
- Claims registry: `docs/project/CLAIMS_PRACTICING_FILES.txt`
