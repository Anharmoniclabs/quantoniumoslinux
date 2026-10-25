# PATENT-PRACTICING FILES LICENSE (LICENSE-CLAIMS-NC)

**Version 2.2 — March 24, 2026**

This license applies to:
1. Files explicitly listed in `docs/project/CLAIMS_PRACTICING_FILES.txt` ("Covered Files")
2. All files bearing `SPDX-License-Identifier: LicenseRef-SPX-Proprietary` ("SPX Files") — governed exclusively by `LICENSE-SPX-PROPRIETARY.md`
3. Research-sandbox files bearing `SPDX-License-Identifier: AGPL-3.0-or-later` ("Research Sandbox Files") — governed by AGPL-3.0 subject to the carve-outs in Section 2 below

All files implement or validate algorithms claimed in **U.S. Patent Application 19/169,399**  
("Hybrid Computational Framework for Quantum and Resonance Simulation").

> **Key Claimed Algorithm**: Hybrid FFT/RFT Transform  
> Formula: Y = E ⊙ FFT(x) / √N where E[k] = e^{i·2π·frac((k+1)φ)}  
> See: `docs/HYBRID_FFT_RFT_ALGORITHM.md`

---

## PREAMBLE

This dual-license structure is designed to:
1. **Protect IP:** Ensure commercial exploitation requires explicit authorization from the independent inventor.
2. **Enable Research:** Allow unrestricted academic verification and reproducibility.
3. **Support Open Science:** Maintain transparency for peer review and validation.

---

## 1. DEFINITIONS

- **"Covered Files"**: Files listed in `docs/project/CLAIMS_PRACTICING_FILES.txt`
- **"SPX Files"**: Files bearing `SPDX-License-Identifier: LicenseRef-SPX-Proprietary` (governed exclusively by `LICENSE-SPX-PROPRIETARY.md`, NOT by this document)
- **"Research Sandbox Files"**: Files bearing `SPDX-License-Identifier: AGPL-3.0-or-later` — see explicit list in Section 2.1
- **"Patent Claims"**: Methods described in USPTO Application 19/169,399
- **"Non-Commercial Use"**: Use that does not generate revenue, competitive advantage, or commercial products
- **"Research Use"**: Scientific inquiry, validation, benchmarking, and peer review
- **"Commercial Use"**: Any use intended for profit, competitive advantage, or integration into commercial products/services
- **"Derivative Work"**: Any work based upon or incorporating the Covered Files

---

## 2. GRANT OF LICENSE (Non-Commercial Research — AGPL Scope)

### 2.1 AGPL Scope — Strictly Limited to Research Sandbox Files

The AGPL-3.0-or-later license in this repository applies **ONLY** to the
following explicitly named files:

**Root-level research sandbox files:**

| File | Purpose | License |
|------|---------|--------|
| `theorem_e_proof_test.py` | Verifies Theorem E (RIP): δ₈ = 0.1603 < 0.2929 | AGPL-3.0-or-later |
| `run_formal_verification.py` | Runs all 255 formal proof assertions + STRT benchmark | AGPL-3.0-or-later |

**Test harnesses:**

| File | Purpose | License |
|------|---------|--------|
| `tests/test_*.py` | Verification harnesses that run proofs | AGPL-3.0-or-later |

**Examples and benchmarks:**

| File | Purpose | License |
|------|---------|--------|
| `examples/*.py` | Demonstration scripts that reproduce published results | AGPL-3.0-or-later |
| `benchmarks/*.py` | Performance benchmarking scripts | AGPL-3.0-or-later |

**Hardware research scripts:**

| File | Purpose | License |
|------|---------|--------|
| `hardware/generate_hardware_test_vectors.py` | Generates RTL test vectors | AGPL-3.0-or-later |
| `hardware/run_capabilities_test.py` | Runs hardware capability tests | AGPL-3.0-or-later |
| `hardware/run_full_verification.py` | Full hardware verification suite | AGPL-3.0-or-later |
| `hardware/theorem8_concentration_test.py` | Theorem 8 concentration bound test | AGPL-3.0-or-later |
| `hardware/visualize_hardware_results.py` | Hardware result visualization | AGPL-3.0-or-later |
| `hardware/visualize_sw_hw_comparison.py` | SW/HW comparison visualization | AGPL-3.0-or-later |

**The AGPL grant for all files above explicitly does NOT cover, transfer, or dilute rights to:**

- The RFT mathematical formulation (Y = E ⊙ FFT(x)/√N)
- The Gram-normalized basis construction (Φ̃ = Φ(ΦᴴΦ)^{-1/2})
- Theorems 1–9 and all associated proofs and derivations
- The RFTMW spectral-entropy routing architecture
- The Kernel ROM definition and φ-grid sampling scheme
- The Hybrid FFT/RFT algorithm or any variant thereof
- Any file bearing the `LicenseRef-SPX-Proprietary` SPDX identifier
- Any file listed in `docs/project/CLAIMS_PRACTICING_FILES.txt`

Running an AGPL-licensed Research Sandbox File does not grant any rights
to the underlying SPX-licensed algorithms it invokes or validates.

### 2.2 Permitted Non-Commercial Uses (Research Sandbox Files Only)

| Use Case | Permitted | Notes |
|----------|-----------|-------|
| **Academic Research** | ✅ YES | Validate claims, reproduce benchmarks, publish findings |
| **Peer Review** | ✅ YES | Verify correctness for journal/conference review |
| **Education** | ✅ YES | Teaching, coursework, student projects |
| **Personal Learning** | ✅ YES | Non-profit hobbyist experimentation |
| **Open-Source Contribution** | ✅ YES | Bug fixes, documentation, tests (upstream only) |
| **Benchmarking** | ✅ YES | Compare performance for research papers |
| **Security Audits** | ✅ YES | Identify vulnerabilities (responsible disclosure) |

### 2.3 Rights Granted for Research (Research Sandbox Files Only)

For Research Sandbox Files only, you may:
- **Copy** for research purposes
- **Modify** to test hypotheses or validate claims
- **Run** to reproduce experimental results
- **Publish** findings, benchmarks, and analyses
- **Share** unmodified files with attribution for collaborative research
- **Create** derivative works for non-commercial research only

These rights do NOT extend to SPX Files or Covered Files.

---

## 3. RESTRICTIONS (Commercial Use Prohibited Without License)

### 3.1 Prohibited Commercial Uses

You may **NOT**, without a separate commercial license:

| Prohibited Use | Description |
|----------------|-------------|
| **Product Integration** | Incorporate algorithms into paid software/hardware |
| **Service Provision** | Use algorithms to process data for paying customers |
| **Competitive Analysis** | Use to build competing commercial products |
| **Resale/Sublicensing** | Sell, license, or sublicense the Covered Files |
| **Patent Circumvention** | Create "clean room" implementations of claimed methods |
| **Hardware Manufacturing** | Fabricate ASICs/FPGAs implementing claimed methods |
| **Cloud Services** | Offer RFT transforms as a service (SaaS/PaaS) |

### 3.2 Patent Rights Reserved

**NO PATENT LICENSE IS GRANTED** for commercial purposes. The algorithms in
Covered Files are subject to:
- **USPTO Application 19/169,399** (pending)
- Any continuation, divisional, continuation-in-part, or foreign counterpart applications

Commercial practice of the claimed methods **requires a separate written patent license** from Luis M. Minier.

### 3.3 Trademark Restrictions

The names "quantoniumos," "RFTPU," "Φ-RFT," "RFTMW," and associated logos
are trademarks of Luis M. Minier and are **NOT licensed** under this agreement.

### 3.4 Patent-Practicing Files — Complete Inventory

The following files implement patented methods and are governed by
`LICENSE-SPX-PROPRIETARY.md`. All commercial use requires a written patent license.

#### Core Algorithm Files — `algorithms/rft/core/`

| File | IP Component | Claims |
|------|-------------|--------|
| `resonant_fourier_transform.py` | Primary patented RFT algorithm | 1–3 |
| `canonical_true_rft.py` | Gram-locked canonical RFT | 1–3 |
| `gram_utils.py` | Gram-normalization kernel | 1 |
| `fast_rft.py` | Fast RFT variant | 1–2 |
| `fast_rft_surrogate.py` | Surrogate fast RFT | 1–2 |
| `fibonacci_fast_rft.py` | Fibonacci-indexed φ-sampled RFT | 1–2 |
| `continuous_compute.py` | Continuous RFT engine | 1–3 |
| `kernel_truncation.py` | Kernel ROM truncation — φ-grid | 2 |
| `sharp_coherence_bounds.py` | Theorem E: δ₈ certificate | 3 |
| `transform_theorems.py` | Theorems 1–9 | 1–5 |

#### RTL/Hardware Files — `hardware/`

| File | IP Component | Claims |
|------|-------------|--------|
| `fpga_top.sv` | 16-mode FPGA top | 1–5 |
| `fpga_top_minimal.sv` | 4-mode minimal FPGA top | 1–3 |
| `fpga_top_webfpga.v` | WebFPGA 4-mode synthesis | 1–3 |
| `rftpu_architecture.sv` | RFTPU full architecture | 1–5 |
| `rftpu_architecture_gen.sv` | RFTPU generated architecture | 1–5 |
| `rftpu_architecture_gen_gen.sv` | RFTPU double-generated RTL | 1–5 |
| `rft_middleware_engine.sv` | RFTMW hardware engine | 1, 4 |
| `tb_rft_middleware.sv` | RFTMW testbench | 1, 4 |
| `fpga_top_tb.v` | FPGA top testbench | 1–3 |
| `full_coverage_tb.v` | Full coverage testbench | 1–5 |
| `testbench.v` | Base testbench | 1–3 |
| `kernel_rom_cases.vh` | Kernel ROM — φ-grid coefficients | 2 |
| `all_kernels.vh` | All-kernel multi-mode ROM | 2 |
| `tlv_multikernel_rom.vh` | TLV multi-kernel ROM | 2 |

#### Engine Files — `quantonium_os_src/engine/`

| File | IP Component | Claims |
|------|-------------|--------|
| `rftmw_memory.py` | Spectral-entropy router, φ-grid RFT compression, KV-cache middleware | 1, 4 |
| `rftmw_inference.py` | Compressed inference engine with on-demand decompression, cryptographic provenance chain | 1, 4 |

---

## 4. ATTRIBUTION REQUIREMENTS

All permitted uses must include:

### 4.1 Source Code Attribution
```
This software includes code from quantoniumos (https://github.com/LMMinier/quantoniumos)
Licensed under LICENSE-CLAIMS-NC.md (Research Use Only)
Patent Pending: USPTO 19/169,399
Copyright (c) 2024-2026 Luis M. Minier
```

### 4.2 Academic Citation
If you publish research using this software, cite:
```bibtex
@software{quantoniumos,
  author = {Minier, Luis M.},
  title = {quantoniumos: Quantum-Inspired Research Platform},
  year = {2025},
  doi = {10.5281/zenodo.17712905},
  note = {Patent Pending: USPTO 19/169,399}
}
```

---

## 5. VERIFICATION AND REPRODUCIBILITY RIGHTS

### 5.1 Special Grant for Scientific Verification

To support open science, this license **explicitly permits**:

1. **Claim Verification**: Running tests to verify mathematical claims (Theorems 1–9)
2. **Benchmark Reproduction**: Reproducing published performance benchmarks
3. **Security Analysis**: Testing cryptographic properties for academic publication
4. **Negative Results**: Publishing findings that contradict claimed properties
5. **Comparison Studies**: Fair comparison with competing transforms/codecs

### 5.2 No Restriction on Criticism

This license does **NOT** restrict your ability to:
- Publish negative or critical findings about the software
- Compare unfavorably to other implementations
- Report bugs, vulnerabilities, or limitations
- Challenge patent validity through proper legal channels

---

## 6. DERIVATIVE WORKS

### 6.1 Non-Commercial Derivatives (Research Sandbox Files Only)

For Research Sandbox Files only, you may create derivative works provided:
- Derivative works inherit this license (LICENSE-CLAIMS-NC)
- Attribution requirements (Section 4) apply
- Commercial use of derivatives remains prohibited
- Derivative works do not implement or expose SPX-protected algorithms

### 6.2 Upstream Contributions

Contributions to the original quantoniumos repository:
- Must be submitted under the Contributor License Agreement (CLA)
- Grant Luis M. Minier rights to include in commercial versions
- Do not grant you commercial rights to the overall work

---

## 7. TERMINATION

### 7.1 Automatic Termination

Your rights under this license terminate automatically if you:
- Use Covered Files commercially without a license
- Use SPX Files beyond view/run-for-verification rights
- Fail to comply with attribution requirements
- Initiate patent litigation related to USPTO 19/169,399
- Violate any other term of this license

### 7.2 Cure Period

For minor violations (e.g., missing attribution), you have **30 days** to cure
after written notice before termination is final.

---

## 8. WARRANTY DISCLAIMER

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY.

---

## 9. COMMERCIAL LICENSING

### 9.1 How to Obtain a Commercial License

For commercial use of patent-practicing code, contact:

**Luis M. Minier**  
📧 Email: luisminier79@gmail.com  
🔗 GitHub: https://github.com/LMMinier/quantoniumos/issues  

### 9.2 Commercial License Tiers

| Tier | Use Case | Contact for Pricing |
|------|----------|---------------------|
| **Startup** | < $1M ARR, < 10 employees | luisminier79@gmail.com |
| **Enterprise** | > $1M ARR | luisminier79@gmail.com |
| **Academic Commercial** | University spin-offs | luisminier79@gmail.com |
| **Hardware** | ASIC/FPGA manufacturing | luisminier79@gmail.com |

---

## 10. GOVERNING LAW

This license is governed by the laws of the **United States** and the
**State of New York**. Any disputes shall be resolved in the federal or
state courts of **New York County, New York**.

---

## 11. NOTICE CROSS-REFERENCE

This repository includes a `NOTICE` file at the root that provides a human-readable
summary of all third-party components used. Third-party components are licensed
separately and do not affect the scope of this license. See `NOTICE` for details.

---

## 12. EXPORT CONTROLS

The algorithms and hardware designs covered by this license may be subject to
U.S. Export Administration Regulations (EAR) or other applicable export control
laws. Recipients are responsible for compliance with all applicable export control
laws and regulations, including obtaining any required export licenses prior to
export, re-export, or transfer of covered technology.

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | February 2026 | Initial public release |
| 2.1 | March 2026 | AGPL scope clarification, RFTMW file table added |
| 2.2 | March 24, 2026 | Added root sandbox files (theorem_e_proof_test.py, run_formal_verification.py) to explicit AGPL scope; added all RTL files to Section 3.4 inventory; added Section 11 (NOTICE cross-reference) and Section 12 (Export Controls); rftpu_architecture_gen_gen.sv added to file tables |

---

**Effective Date:** March 24, 2026  
**Version:** 2.2 (supersedes v2.1 of March 2026)  
**Copyright:** © 2024–2026 Luis M. Minier. All rights reserved.
