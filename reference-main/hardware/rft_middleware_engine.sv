// =============================================================================
// SPDX-License-Identifier: LicenseRef-SPX-Proprietary
// Copyright (c) 2024-2026 Luis M. Minier. All rights reserved.
//
// Patent Pending: USPTO Application 19/169,399 — Claims 1, 4
// "Hybrid Computational Framework for Quantum and Resonance Simulation"
//
// RFT MIDDLEWARE ENGINE — RTL implementation of the RFTMW pipeline (Layer 1)
// ==========================================================================
// STATUS: SIMULATION-ONLY — no silicon, no deployed FPGA bitstream.
//
// This module implements the Spectral-Entropy-Routed Memory Middleware in
// synthesizable SystemVerilog, directly practising Claims 1 and 4 of
// USPTO 19/169,399.
//
// Functional description
// ----------------------
// The RFTMW engine is the hardware counterpart of the software middleware in
// quantonium_os_src/engine/RFTMW.py.  It performs the four-stage pipeline:
//
//   1. Binary → Waveform   : input samples loaded into the φ-grid basis
//   2. Waveform → Compute  : RFT variant selected by the variant_sel[3:0] port
//   3. Compute → Spectral  : entropy-routed coefficient thresholding / routing
//   4. Spectral → Binary   : inverse RFT and output serialization
//
// Variant select port (variant_sel[3:0]) maps to the same 16 modes as
// fpga_top.sv — see that file for the mode-to-variant mapping table.
//
// Clock/reset
// -----------
//   clk    : rising-edge sensitive
//   rst_n  : active-low synchronous reset
//
// PERMITTED:     View for peer review. Run synthesis to verify results.
// NOT PERMITTED: Fabricate, copy, modify, integrate, or use commercially
//                without a written patent license from Luis M. Minier.
//
// Commercial licensing: luisminier79@gmail.com
// See LICENSE-SPX-PROPRIETARY.md for full terms.
// =============================================================================
