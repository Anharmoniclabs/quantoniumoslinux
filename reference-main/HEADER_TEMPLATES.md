# FILE HEADER TEMPLATES

Copy the appropriate header block to the **top of every file** in this repository.
Choose the correct category based on what the file actually does.

---

## CATEGORY 1 — Core Algorithm / Math / Theorems / Architecture

**Use for:** `algorithms/rft/core/theorems/`, `algorithms/rft/core/experimental/`,
Kernel ROM files, Gram basis files, RFTMW architecture files, any file containing
the RFT formula, proof derivations, or novel scheme design.

```python
# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# This file contains proprietary mathematical frameworks, algorithm
# specifications, and/or theorem proofs that are the exclusive intellectual
# property of Luis M. Minier.
#
# PERMITTED: View for peer review and academic verification only.
#            Run to reproduce published experimental results.
# NOT PERMITTED: Copy, modify, redistribute, fork, implement independently,
#                or use commercially without a written patent license.
#
# Commercial licensing: luisminier79@gmail.com
# See LICENSE-SPX-PROPRIETARY.md for full terms.
# =============================================================================
```

For **Verilog/SystemVerilog RTL files**, use:
```verilog
// =============================================================================
// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
//
// Patent Pending: USPTO Application 19/169,399
// "Hybrid Computational Framework for Quantum and Resonance Simulation"
//
// This RTL file implements hardware architecture claimed in USPTO 19/169,399.
// PERMITTED: View for peer review. Run synthesis to verify published results.
// NOT PERMITTED: Fabricate, copy, modify, or use commercially without a
//                written patent license from Luis M. Minier.
//
// Commercial licensing: luisminier79@gmail.com
// See LICENSE-SPX-PROPRIETARY.md for full terms.
// =============================================================================
```

---

## CATEGORY 2 — Research Sandbox (Tests / Examples / Benchmarks)

**Use for:** All files in `tests/`, `examples/`, `benchmarks/`.
These files run the system to verify claims but do NOT implement the core algorithms.

```python
# =============================================================================
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Luis M. Minier.
#
# Research Sandbox File — Scientific Verification & Reproducibility Only
#
# This file is licensed under the GNU Affero General Public License v3.0
# or later FOR RESEARCH AND SCIENTIFIC PURPOSES ONLY.
#
# The AGPL grant covers THIS FILE ONLY. It does NOT transfer any rights
# to the underlying algorithms, mathematical formulations, or architectural
# designs this file validates. Those are protected by:
#   - USPTO Application 19/169,399 (Patent Pending)
#   - LICENSE-SPX-PROPRIETARY.md (LicenseRef-SPX-Proprietary)
#
# You may: run, modify, and publish results from this file for academic research.
# You may NOT: use the validated algorithms commercially without a patent license.
#
# Full terms: LICENSE-CLAIMS-NC.md Section 2
# Patent licensing: luisminier79@gmail.com
# =============================================================================
```

---

## CATEGORY 3 — Patent-Practicing Engine Files

**Use for:** `quantonium_os_src/engine/rftmw_memory.py`,
`quantonium_os_src/engine/rftmw_inference.py`, and any engine file that
directly implements a claimed method.

```python
# =============================================================================
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
# Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
#
# Patent Pending: USPTO Application 19/169,399 — Claims 1, 4
# "Hybrid Computational Framework for Quantum and Resonance Simulation"
#
# This file implements the Spectral-Entropy-Routed Memory Middleware (RFTMW)
# which is directly claimed in USPTO Application 19/169,399.
#
# NOTICE: This file is NOT covered by the AGPL-3.0 license. The AGPL grant
# in this repository applies only to tests/, examples/, and benchmarks/.
#
# PERMITTED: View for peer review. Run for scientific result reproduction.
# NOT PERMITTED: Copy, integrate into products, offer as a service, or
#                use commercially without a written patent license.
#
# Commercial licensing: luisminier79@gmail.com
# Full terms: LICENSE-SPX-PROPRIETARY.md and LICENSE-CLAIMS-NC.md
# =============================================================================
```

---

## CATEGORY 4 — Utilities / Infrastructure (No Novel IP)

**Use for:** `utils/bloom_filter.py`, `utils/shard.py`, `utils/oscillator.py`,
`atomic_io.py`, scripts, Dockerfiles, CI config, and any file containing
no novel algorithm IP.

```python
# =============================================================================
# SPDX-License-Identifier: MIT
# Copyright (c) 2024-2026 Luis M. Minier.
#
# Utility infrastructure file — no novel algorithm IP.
# This file does not implement or expose any methods claimed in
# USPTO Application 19/169,399.
#
# See LICENSE for full MIT terms.
# =============================================================================
```

---

## QUICK REFERENCE — Which Header Goes Where

| Directory / File Pattern | Header Category |
|---|---|
| `algorithms/rft/core/theorems/` | Category 1 — SPX Proprietary |
| `algorithms/rft/core/experimental/` | Category 1 — SPX Proprietary |
| `algorithms/rft/core/legacy/` | Category 1 — SPX Proprietary |
| `hardware/*.sv`, `hardware/*.v` | Category 1 — SPX Proprietary (RTL variant) |
| `quantonium_os_src/engine/rftmw_*.py` | Category 3 — Patent-Practicing Engine |
| `quantonium_os_src/engine/*.py` (other) | Category 1 — SPX Proprietary |
| `tests/test_*.py` | Category 2 — AGPL Research Sandbox |
| `examples/*.py` | Category 2 — AGPL Research Sandbox |
| `benchmarks/*.py` | Category 2 — AGPL Research Sandbox |
| `theorem_e_proof_test.py` (root) | Category 2 — AGPL Research Sandbox |
| `algorithms/rft/core/utils/` | Category 4 — MIT Utility |
| `atomic_io.py`, `cleanup_repo.sh`, etc. | Category 4 — MIT Utility |
| `scripts/`, `tools/` | Category 4 — MIT Utility |

---

*Generated: March 24, 2026 — Luis M. Minier / quantoniumos*
*Patent Pending: USPTO 19/169,399*
