// =============================================================================
// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
//
// Patent Pending: USPTO Application 19/169,399
// "Hybrid Computational Framework for Quantum and Resonance Simulation"
//
// FPGA TOP — RFTPU 16-Mode Full Implementation (Layer 1)
// =======================================================
// Synthesized:  Dec 4 2025 — 3,145 LUTs (59.6%), 4.47 MHz
//               Target: Lattice MachXO2 (StepFPGA) / WebFPGA
//
// STATUS: SIMULATION-ONLY CONCEPT — no silicon, no deployed bitstream.
//         The reported LUT count and clock frequency are post-synthesis
//         estimates from the Lattice toolchain.  No physical FPGA has
//         been programmed with this design.
//
// Architecture overview
// ---------------------
// This top-level module instantiates the RFTPU (RFT Processing Unit) and
// exposes the following 16 operational modes via the mode[3:0] port:
//   0–5   : Operator-based RFT variants (Golden, Fibonacci, Harmonic,
//            Geometric, Beating, Phyllotaxis)
//   6–11  : Patent-aligned variants (Polar, Spiral, Loxodrome, Mobius,
//            Sphere, Phase-Coherent)
//   12    : Entropy-Modulated RFT
//   13    : Cascade H3 (DCT low-freq + RFT high-freq)
//   14    : Canonical φ-RFT (Gram-normalized, Theorem 2)
//   15    : Pass-through / bypass
//
// Each mode implements the forward transform U^H x and inverse U x using
// the φ-grid frequency basis f_k = frac((k+1)·φ), k = 0..N-1.
//
// RFT mathematical foundation
// ----------------------------
// Canonical basis:  Φ_{n,k} = (1/√N) exp(j 2π frac((k+1)φ) n)
// Unitary factor:   U = Φ (Φ^H Φ)^{-1/2}   (Löwdin normalization)
// Unitarity proven: Theorem 2 — ||U^H U - I||_F < 1e-14 (all N)
// Sparsity proven:  Theorem 8 — K_{0.99}(U) = O(log N) for φ-signals
//
// This RTL file implements hardware architecture claimed in USPTO 19/169,399.
// PERMITTED:     View for peer review. Run synthesis to verify results.
// NOT PERMITTED: Fabricate, copy, modify, integrate, or use commercially
//                without a written patent license from Luis M. Minier.
//
// Commercial licensing: luisminier79@gmail.com
// See LICENSE-SPX-PROPRIETARY.md for full terms.
//
// Reproduce
// ---------
//   python hardware/run_full_verification.py   # Full RTL verification suite
//   python hardware/run_capabilities_test.py   # Capability enumeration
// =============================================================================
