# SPX PROPRIETARY LICENSE

**SPDX-License-Identifier: LicenseRef-SPX-Proprietary**

**Version 1.1 — March 24, 2026**  
**Copyright © 2024–2026 Luis M. Minier. All rights reserved.**

---

## SCOPE

This license governs ALL files bearing the SPDX header:

```
# SPDX-License-Identifier: LicenseRef-SPX-Proprietary
```

or for RTL/hardware:

```
// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
```

### Covered Intellectual Property

This includes, but is not limited to, all files implementing or describing:

- The **Resonance Fourier Transform (RFT)** mathematical formulation and all variants
- The **Gram-normalized basis** construction: Φ̃ = Φ(ΦᴴΦ)^{-1/2}
- The **Hybrid FFT/RFT Transform** formula: Y = E ⊙ FFT(x)/√N, where E[k] = e^{i·2π·frac((k+1)φ)}
- **Theorems 1–9** and all associated mathematical proofs and derivations
- The **RFTMW Spectral-Entropy-Routed Memory Middleware** architecture
- The **Kernel ROM** definition and φ-grid sampling scheme
- All algorithm architecture documents, design specifications, and scheme descriptions
- All hardware RTL files implementing the above algorithms

### Explicit File Inventory (Non-Exhaustive)

#### Python Source Files — `algorithms/rft/core/`

| File | IP Component |
|------|--------------|
| `resonant_fourier_transform.py` | Primary patented algorithm (Claims 1–3) |
| `canonical_true_rft.py` | Gram-locked canonical RFT |
| `gram_utils.py` | Gram-normalization kernel Φ̃ = Φ(ΦᴴΦ)^{-1/2} |
| `fast_rft.py` | Fast RFT variant |
| `fast_rft_surrogate.py` | Surrogate fast RFT |
| `fibonacci_fast_rft.py` | Fibonacci-indexed φ-sampled RFT |
| `continuous_compute.py` | Continuous RFT computation engine |
| `unified_continuous.py` | Unified continuous transform |
| `true_wave_compute.py` | Wave-domain computation |
| `symbolic_wave_computer.py` | Symbolic wave computation |
| `sharp_coherence_bounds.py` | Theorem E: δ₈ = 0.1603 < 0.2929 |
| `maassen_uffink_uncertainty.py` | Quantum uncertainty bounds |
| `golden_uncertainty_principle.py` | Golden-ratio uncertainty principle |
| `golden_reservoir.py` | Golden reservoir sampling |
| `diophantine_rft_extension.py` | Diophantine RFT extension |
| `kernel_truncation.py` | Kernel ROM truncation — φ-grid |
| `absolute_novelty.py` | DFT distance certificate |
| `rft_phi_legacy.py` | Legacy pre-Gram-locked RFT |
| `oscillator.py` | Resonance oscillator core |
| `vibrational_engine.py` | Vibrational engine |
| `geometric_container.py` | Geometric container |
| `shard.py` | Sharding utility |
| `bloom_filter.py` | Bloom filter for RFT indexing |
| `theorem8_bootstrap_verification.py` | Theorem 8 bootstrap |
| `transform_theorems.py` | Theorems 1–9 derivations |
| `__init__.py` | Package interface |

#### RTL / Hardware Files — `hardware/`

| File | IP Component | Claims |
|------|-------------|--------|
| `fpga_top.sv` | Full 16-mode FPGA top | 1–5 |
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
| `all_kernels.vh` | All-kernel ROM | 2 |
| `tlv_multikernel_rom.vh` | TLV multi-kernel ROM | 2 |

#### Engine Files — `quantonium_os_src/engine/`

| File | IP Component | Claims |
|------|-------------|--------|
| `rftmw_memory.py` | Spectral-entropy router, φ-grid compression, KV-cache middleware | 1, 4 |
| `rftmw_inference.py` | Compressed inference engine | 1, 4 |

---

## GRANT OF RIGHTS

Subject to this license, Luis M. Minier grants you the following **limited rights only**:

| Right | Permitted | Conditions |
|-------|-----------|------------|
| **View / Read** | ✅ YES | For peer review and academic verification only |
| **Run (verification)** | ✅ YES | To reproduce published results for scientific validation |
| **Cite in publications** | ✅ YES | With attribution per Section 4 of LICENSE-CLAIMS-NC.md |
| **Copy** | ❌ NO | Not permitted without written commercial license |
| **Modify** | ❌ NO | Not permitted without written commercial license |
| **Redistribute** | ❌ NO | Not permitted under any circumstances |
| **Independent implementation** | ❌ NO | See clean-room prohibition below |
| **Fork** | ❌ NO | Not permitted without written commercial license |
| **Sublicense** | ❌ NO | Not permitted |
| **Commercial use** | ❌ NO | Requires separate written patent license |
| **Hardware fabrication** | ❌ NO | ASIC/FPGA manufacture requires written patent license |

---

## PATENT NOTICE

The algorithms, mathematical frameworks, and architectural designs in files
bearing this identifier are subject to:

**U.S. Patent Application 19/169,399**  
"Hybrid Computational Framework for Quantum and Resonance Simulation"  
Inventor: Luis M. Minier  
Filing Status: Pending

And any continuation, divisional, continuation-in-part, or foreign counterpart
applications filed subsequently.

**NO PATENT LICENSE IS GRANTED** under this agreement for any purpose.  
Commercial practice of any claimed method requires a separate written patent
license from Luis M. Minier.

---

## CLEAN-ROOM PROHIBITION

The creation of a "clean-room" or independent re-implementation of any algorithm,
formula, architecture, or mathematical framework protected by this license or
by USPTO Application 19/169,399 is **expressly prohibited** without a written
commercial patent license from Luis M. Minier.

This prohibition applies regardless of whether the implementing party has
directly accessed these files, consistent with patent law principles establishing
that knowledge of a claimed invention (including through public disclosure of a
pending patent application) bars independent implementation without a license.

Specifically prohibited without a license:

- Implementing Y = E ⊙ FFT(x)/√N with E[k] = e^{i·2π·frac((k+1)φ)}
- Implementing Φ̃ = Φ(ΦᴴΦ)^{-1/2} Gram normalization for transform bases
- Implementing Spectral-Entropy-Routed Memory Middleware (RFTMW)
- Implementing φ-grid Kernel ROM sampling schemes
- Fabricating RTL implementing any of the above

---

## DMCA / ANTI-CIRCUMVENTION

Any technical measure implemented to circumvent access controls or license
enforcement mechanisms in this repository is prohibited under 17 U.S.C. § 1201
(Digital Millennium Copyright Act) and applicable international law.

---

## RELATIONSHIP TO AGPL

The GNU Affero General Public License (AGPL-3.0-or-later) that applies to
certain research-sandbox files in this repository **does NOT apply to files
bearing this SPX identifier**.

The AGPL grant in this repository is explicitly scoped to research
reproducibility harnesses only. It does **not** transfer, dilute, or grant
any rights to:

- The mathematical formulations implemented by those harnesses
- The algorithms called by those harnesses
- The architectural designs tested by those harnesses
- Any intellectual property protected by this SPX license or USPTO 19/169,399

Running an AGPL-licensed test file does not grant you any rights to the
underlying SPX-licensed algorithm it validates.

---

## EXPORT CONTROLS

The algorithms and hardware designs covered by this license may be subject
to U.S. Export Administration Regulations (EAR) or other applicable export
control laws. Recipients are responsible for compliance with all applicable
export control laws and regulations, including obtaining any required export
licenses prior to export, re-export, or transfer of covered technology.

---

## TRADEMARKS

The following are trademarks of Luis M. Minier and are NOT licensed under
any open-source or research license in this repository:

- **quantoniumos**
- **RFTPU**
- **Φ-RFT**
- **RFTMW**

Use of these marks requires separate written trademark authorization.

---

## DISCLAIMER

THE COVERED FILES ARE PROVIDED FOR PEER REVIEW AND SCIENTIFIC VERIFICATION
ONLY. THEY ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. IN NO EVENT
SHALL LUIS M. MINIER BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY
ARISING FROM ACCESS TO OR USE OF THESE FILES.

---

## COMMERCIAL LICENSING

To obtain a commercial license covering files under this SPX identifier:

**Luis M. Minier**  
📧 luisminier79@gmail.com  
🔗 https://github.com/LMMinier/quantoniumos/issues  

---

## VERSION HISTORY

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | March 2026 | Initial release |
| 1.1 | March 24, 2026 | Added explicit file inventory (Python + RTL), clean-room prohibition clause, DMCA notice, export control section |

**Effective Date:** March 24, 2026  
**Copyright:** © 2024–2026 Luis M. Minier. All rights reserved.
