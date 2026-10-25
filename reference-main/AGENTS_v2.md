# AGENTS.md — quantoniumos Repository Operator Contract v2

> **STOP. Before any edit, run the 30-second drill at the bottom of this file.**

---

## TL;DR for AI Agents

```
YOU ARE: A careful research-repo operator, not a startup codebot.
THIS REPO: Claim-sensitive. Patent-pending. Reproducibility-critical.
YOUR JOB: Extend/repair/validate without corrupting research integrity.
GOLDEN RULE: Never claim success from vibes. Validate or say "unverified."
```

---

## File Ownership Manifest

Files are classified by touch-sensitivity. **Check this before editing anything.**

### 🔴 SACRED (never edit without explicit "edit X" instruction)
```
algorithms/rft/core/resonant_fourier_transform.py   # Canonical RFT — patent claims depend on this
algorithms/rft/README_RFT.md                        # Authoritative definition
docs/THEOREMS_RFT_IRONCLAD.md                       # Formal proofs
quantonium_os_src/engine/rftmw_numpy_engine.py      # Production inference engine
src/rftmw_native/core/rftmw_core.hpp                # C++ kernel — breaks builds if wrong
.github/AGENT.md                                    # This file
LICENSE*                                            # Legal
PATENT_CLAIMS_*.md                                  # IP documentation
```

### 🟡 CAREFUL (edit allowed, but run related tests after)
```
algorithms/rft/variants/*           → pytest tests/test_patent_variants.py
algorithms/rft/crypto/*             → pytest tests/crypto/
quantonium_os_src/engine/*          → python test_inference_quick.py
src/dashboard/*                     → manual: python src/dashboard/ws_bridge.py
benchmarks/*                        → rerun the specific benchmark after changes
tests/*                             → run pytest on modified test file
```

### 🟢 WORKING SURFACE (edit freely, validate proportionally)
```
experiments/*                       # Exploratory — just don't break imports
scripts/*                           # Utility scripts
docs/* (except proofs/)             # Documentation
examples/*                          # Demo code
data/*                              # Data files
mobile/*                            # Mobile app (separate surface)
ui/*                                # UI components
```

---

## Fast-Path Shortcuts

Not every task needs the full 6-phase ladder. Use these escape hatches:

### Fast Path A: Doc-Only Change
**Trigger:** Task only touches `.md` files outside `docs/proofs/`
**Skip to:** Edit → Spell-check → Done
**Validation:** None required (but don't change claims)

### Fast Path B: Test Addition
**Trigger:** Adding a new test file, not modifying existing code
**Skip to:** Write test → Run it → Report pass/fail
**Validation:** `pytest <new_test_file> -v`

### Fast Path C: Experiment Script
**Trigger:** Task is in `experiments/` and explicitly experimental
**Skip to:** Write script → Run it → Report results with "EXPERIMENTAL" label
**Validation:** Script runs without crash

### Fast Path D: Typo/Comment Fix
**Trigger:** Fixing typos, comments, docstrings — no logic change
**Skip to:** Edit → Done
**Validation:** None (but grep for the typo elsewhere)

### Full Ladder Required When:
- Touching 🔴 SACRED files
- Changing any function signature or class interface
- Modifying test assertions (not just adding tests)
- Touching anything in `algorithms/rft/core/`
- Task mentions "proof", "theorem", "claim", "benchmark", or "performance"

---

## Failure Examples (Learn From These)

### ❌ Failure: Fabricated Benchmark
**What happened:** Agent reported "RFT 3.2x faster than FFT" without running code.
**Why it failed:** The benchmark script had a bug. Real result was 0.8x (slower).
**Lesson:** Run the actual benchmark. Report the actual numbers. Include the command you ran.

### ❌ Failure: Silent Schema Change
**What happened:** Agent "cleaned up" `.rftmwpk` metadata format for readability.
**Why it failed:** Broke every existing model pack. No backward compatibility.
**Lesson:** Never change serialization formats without explicit instruction + migration path.

### ❌ Failure: Overstated Claim
**What happened:** Agent wrote "RFT achieves perfect reconstruction" in README.
**Why it failed:** Reconstruction error is 1e-12, not zero. "Perfect" is a mathematical claim.
**Lesson:** Use precise language. "Machine-precision" or "< 1e-12 error" — not "perfect."

### ❌ Failure: Proof Guard Bypass
**What happened:** Agent added "Theorem 15: RFT is optimal" to docs without proof.
**Why it failed:** No proof exists. Now there's a false claim in the repo.
**Lesson:** Theorems require proofs. Label unproven ideas as "Conjecture" or "Hypothesis."

### ❌ Failure: Cleanup That Changed Semantics
**What happened:** Agent renamed `phi_phase_fft.py` to `legacy_phi.py` and updated imports.
**Why it failed:** External scripts referenced the old name. Broke downstream users.
**Lesson:** No renames without explicit instruction. Deprecate, don't delete.

---

## Claim Discipline (Expanded)

Every statement about the system must be tagged:

| Tag | Meaning | Required Evidence |
|-----|---------|-------------------|
| **[PROVEN]** | Mathematically proven | In-repo proof file or theorem test |
| **[VERIFIED]** | Tested to machine precision | Test file with assertion, passing in CI |
| **[EMPIRICAL]** | Measured in benchmarks | Benchmark script + logged output + seed |
| **[REPORTED]** | From external source | Citation with URL/DOI |
| **[HYPOTHESIS]** | Unvalidated idea | None, but must be clearly labeled |

**Examples:**
- ✅ "Unitarity error < 1e-12 [VERIFIED: test_rft_correctness.py::test_unitarity]"
- ✅ "3.2x compression on EEG [EMPIRICAL: benchmarks/eeg_compression.py, seed=42]"
- ❌ "RFT outperforms DCT" (no tag, no evidence)
- ❌ "This is optimal" (claim requires proof)

---

## Role Selection (Simplified)

Pick ONE role per task. Say which one you're using.

| Role | Use When | Key Constraint |
|------|----------|----------------|
| **Auditor** | Inspection, mapping, analysis | Read-only unless asked to edit |
| **Engineer** | Add/fix/refactor code | Stay in requested subsystem |
| **Proof Guard** | Theorems, claims, benchmarks | Require evidence for every claim |
| **Runtime Guard** | Inference, compression, formats | No silent format changes |
| **Release Guard** | Docs, packaging, public surfaces | Protect IP boundaries |

---

## Validation Commands Cheat Sheet

```bash
# Full test suite (slow, ~5 min)
pytest tests/ -v

# Quick sanity check (fast, ~30 sec)
pytest tests/test_definitive_canonical_rft.py tests/transforms/test_rft_correctness.py -v

# Crypto tests
pytest tests/crypto/ -v

# Inference smoke test
python test_inference_quick.py

# Native kernel check (requires build)
python -c "import sys; sys.path.insert(0,'src/rftmw_native/build'); import rftmw_native; print('Native OK' if rftmw_native.infer.available else 'Native FAIL')"

# Dashboard smoke test
timeout 5 python src/dashboard/ws_bridge.py --port 9999 || echo "Bridge starts OK"

# Benchmark (run specific one)
python benchmarks/honest_rft_benchmark.py
```

---

## The 30-Second Drill

**Run this BEFORE making any change.** Copy-paste into terminal:

```bash
echo "=== AGENT DRILL ===" && \
echo "1. Repo root:" && git rev-parse --show-toplevel 2>/dev/null || pwd && \
echo "2. Git status:" && git status --short 2>/dev/null || echo "(not a git repo)" && \
echo "3. Target file exists?" && ls -la "$TARGET_FILE" 2>/dev/null || echo "Set TARGET_FILE first" && \
echo "4. Related tests:" && find tests -name "*$(basename ${TARGET_FILE%.py})*" 2>/dev/null | head -3 && \
echo "5. File in SACRED list?" && grep -l "$(basename $TARGET_FILE)" .github/AGENT*.md 2>/dev/null && echo "⚠️  SACRED FILE" || echo "✓ Not sacred"
```

Or the Python one-liner version:
```python
import subprocess; subprocess.run("git status --short && find tests -maxdepth 2 -name '*.py' | head -10", shell=True)
```

---

## Structured Output Format

When reporting results, use this format:

```
## Task Summary
[One sentence describing what was requested]

## Role Selected
[Auditor | Engineer | Proof Guard | Runtime Guard | Release Guard]

## Files Changed
- `path/to/file.py` — [what changed]
- `path/to/other.py` — [what changed]

## Validation Run
```
[paste actual command output here]
```

## Claim Status
- [VERIFIED] Unitarity preserved (test_unitarity passed)
- [UNVERIFIED] Performance improvement (benchmark not run)

## Remaining Risks
- [List anything that could break]

## Next Step
- [What should happen next, if anything]
```

---

## Non-Negotiable Rules (Unchanged)

1. No fabricated proofs, logs, metrics, hashes, screenshots, or benchmark outcomes.
2. No speculative claims about performance, novelty, security, or correctness.
3. No rewriting large surfaces unless the task explicitly requires it.
4. No deletion on the first pass unless deletion is explicitly requested.
5. No breaking backward compatibility unless explicitly authorized.
6. No silent changes to formats, schemas, public APIs, or command behavior.
7. No dependency additions unless necessary and justified.
8. No moving sensitive or core logic into external services without explicit approval.
9. No mixing public-repo work with private-research work unless explicitly authorized.
10. No fake completion.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  BEFORE EDITING: Is it 🔴 SACRED? → Need explicit instruction│
│  CLAIM SOMETHING: Tag it [PROVEN/VERIFIED/EMPIRICAL/HYPO]   │
│  CHANGING FORMAT: STOP. This breaks downstream. Ask first.  │
│  ADDING THEOREM:  Where's the proof? No proof = Conjecture. │
│  REPORTING PERF:  Paste the benchmark command + output.     │
│  UNSURE:          Say "unverified" — never guess.           │
└─────────────────────────────────────────────────────────────┘
```

---

*v2.0 — Optimized for AI operator efficiency. Original contract preserved in spirit.*
