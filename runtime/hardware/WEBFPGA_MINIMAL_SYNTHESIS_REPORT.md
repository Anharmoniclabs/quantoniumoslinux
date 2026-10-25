# quantoniumos WebFPGA Minimal Synthesis Report

**Date:** March 24, 2026  
**Target:** WebFPGA (Lattice iCE40UP5K)  
**Tool:** WebFPGA Cloud Synthesis (`shastaplus`)  
**Source:** `fpga_top_webfpga.v` (4-mode minimal, Verilog-2001)  
**Stream ID:** `2e6755e84c96b4981968cbd4f32e62ed73b662dae4bca20393d1e50a6a04268d`

> ⚠️ **Scope note:** This is the **reduced 4-mode version** for reliable WebFPGA cloud synthesis.  
> The full 16-mode version is `hardware/fpga_top.sv` — see `WEBFPGA_SYNTHESIS_REPORT.md` for that run.  
> No physical board was flashed during this session (USB permission blocked by browser security policy — see note below).

---

## Synthesis Results ✓ PASSED

| Metric | Value | Utilization |
|--------|-------|-------------|
| **LUT4s** | 741 | 14.03% |
| **Flip-Flops** | 205 | 3.88% |
| **I/Os** | 10 | 25.64% |
| **Logic Cells** | 741 | — |
| **Nets** | 881 | — |
| **Routing Iterations** | 4 | — |

### Timing Analysis

| Clock Domain | Achieved | Target | Status |
|--------------|----------|--------|--------|
| `fpga_top\|WF_CLK` | **33.33 MHz** | 1.00 MHz | ✓ 33.33× margin |

> **Placer intermediate result:** 46.69 MHz; static timing conservatively reports **33.33 MHz** as the signed-off frequency.

---

## Integrated Engines (4 Verified Modes)

This minimal build implements only the four highest-priority verified modes:

| Mode | Engine | Class | Description |
|------|--------|-------|-------------|
| 0 | `RFT_GOLDEN` | 🟢 Class B | Canonical Gram-normalized Φ-RFT basis (unitarity: 4.56e-15) |
| 1 | `RFT_CASCADE` | 🟢 Class B Winner | H3 Hybrid Compression (0.673 BPP, η=0) |
| 2 | `SIS_HASH` | 🟢 Class D | Lattice-based post-quantum hash (Q=3329) |
| 3 | `QUANTUM_SIM` | 🟢 Class A | GHZ symbolic engine — classical fixed-point register representation of (|000⟩ + |111⟩)/√2 |

**Kernel ROM:** 4 modes × 64 coefficients = **256 entries** (vs. 768 in full 16-mode build).  
**Fixed-point format:** Q1.15 (16-bit signed).  
**All kernels:** unitarity error < 5e-15 (machine precision).

### Mode 0 — RFT-GOLDEN (Canonical)

The January 2026 canonical basis: **Φ̃ = Φ(ΦᴴΦ)^{-1/2}** where Φ[n,k] = exp(j 2π frac((k+1)φ) n) / √N.  
Gram-orthonormalized. Unitarity error: **4.56e-15**.

### Mode 3 — QUANTUM_SIM (Symbolic GHZ)

Implements a **classical symbolic representation** of the GHZ state (|000⟩ + |111⟩)/√2 as a fixed-point kernel pattern.  
The kernel places amplitude 23170 (= 0.707 × 32768 in Q1.15) at positions [0,0], [0,7], [7,0], [7,7] and zero elsewhere — encoding the two non-zero amplitudes of the GHZ superposition.  
**This is not a quantum circuit simulator.** It is a hardware-accelerated symbolic fixed-point engine.  
LED output: bits 0 and 7 lit (`8'b10000001`) to visually represent the two GHZ basis states.

---

## Hardware Parameters

### Fixed-Point Format: Q1.15

| Constant | Q1.15 Value | Decimal |
|----------|-------------|---------|
| 1/√2 (GHZ amplitude) | 23170 | 0.70711 |
| Normalization scale | 11585 | 0.35355 |

### Pin Mapping (iCE40UP5K WebFPGA PCF)

```
set_io WF_LED    31
set_io WF_CLK    35
set_io WF_BUTTON 42
set_io WF_NEO    32
set_io WF_CPU1   11
set_io WF_CPU2   12
set_io WF_CPU3   13
set_io WF_CPU4   10
```

---

## LED Behavior by Mode

| Mode | Engine | LED Display |
|------|--------|-------------|
| 0 — RFT_GOLDEN | Φ-RFT | Frequency bin amplitudes (threshold > 0x10) |
| 1 — RFT_CASCADE | H3 | Frequency bin amplitudes (threshold > 0x10) |
| 2 — SIS_HASH | Lattice Hash | `rft_out[0][7:0] ^ rft_out[7][7:0]` |
| 3 — QUANTUM_SIM | Symbolic GHZ | `8'b10000001` (LEDs 0 & 7 lit) |
| Computing | Any | Current mode in binary on LEDs [1:0] |

**Auto-cycle:** ~1.4 s per mode (24-bit counter overflow at 12 MHz).  
**Button:** Manual advance to next mode.

---

## Synthesis Log (Raw)

```
Subscribed to stream: 2e6755e84c96b4981968cbd4f32e62ed73b662dae4bca20393d1e50a6a04268d
synthesis-engine has received the request...
Saving top.v...
Synthesizing...

found the top-level module
====> fpga_top (file /tmp/synthesis_worker.bec44816-bf94-4e46-9690-67e2d2b60901.20260324-20-omsjce/top.v) <====
{
  "top_level": "fpga_top",
  "io_map": {},
  "top_level_found": true,
  "error": false
}

Top level verilog module is fpga_top.

Synthesis:Verilog to EDIF: LUT4s:739(14.00%)
                           FLOPs:205( 3.88%)
                           IOs  :10(25.64%)

EDIF Parser: Successful.
PLACER: Run time up to 2:00 minutes. Compute intensive.

Placement of Design:       LUT4s:741(14.03%)
                           FLOPs:205( 3.88%)
                           IOs:  10(25.64%)
                           CLOCK:fpga_top|WF_CLK  FREQ: 46.69 MHz
                                       target: 1.00 MHz

Packer: DRC check was successful. Now doing packing.
Packer: Packing was successful. Logic cells used: 741

Routing of Design: Successful. Nets:881 Iterations:4.
Writing of Design Netlist: Successful.

Static Timing of Design:
Timing completed. See timing report file.
Clock: fpga_top|WF_CLK  | Frequency: 33.33 MHz  | Target: 1.00 MHz

Bit File Generation: Successful.
header  : E+Tue Mar 24 02:26:12 PM UTC 2026+shastaplus
header2 : E+Tue_Mar_24_02:26:12_PM_UTC_2026+shastaplus

synthesis complete!
```

---

## USB Flash Status

> ❌ **Bitstream NOT flashed to hardware.**

The WebFPGA browser interface reported:
```
SecurityError: Failed to execute 'requestDevice' on 'USB':
Must be handling a user gesture to show a permission request.
error: unable to enumerate USB device list
```

The bitstream was generated successfully and loaded in the browser, but the USB WebUSB permission dialog requires a direct user click gesture to proceed. The board was **not programmed** during this session.  
To flash: open WebFPGA in browser, load the bitstream, then click the **Flash** button directly.

---

## Resource Comparison: Minimal vs. Full Build

| Resource | Minimal (4-mode) | Full (16-mode, Dec 2025) | iCE40UP5K Available |
|----------|-----------------|--------------------------|---------------------|
| LUT4s | 741 (14.03%) | 3,145 (59.56%) | 5,280 |
| FLOPs | 205 (3.88%) | 873 (16.53%) | 5,280 |
| I/Os | 10 (25.64%) | 10 (25.64%) | 39 |
| Achieved Freq | 33.33 MHz | 4.47 MHz | — |
| Kernel ROM entries | 256 | 768 | — |

The minimal build achieves **7.5× higher clock frequency** and uses **76% fewer LUTs** by restricting to the 4 verified production modes, making it the preferred target for reliable WebFPGA flashing.

---

## Source Files

| File | Description |
|------|-------------|
| `hardware/fpga_top_webfpga.v` | This synthesis source — 4-mode minimal, Verilog-2001 |
| `hardware/fpga_top.sv` | Full 16-mode build (SystemVerilog) |
| `hardware/WEBFPGA_SYNTHESIS_REPORT.md` | Dec 4 2025 full-build synthesis report |

---

## Patent & Licensing

- **SPDX License:** `LicenseRef-quantoniumos-Claims-NC`
- **Copyright:** © 2025–2026 Luis M. Minier / quantoniumos
- **Patent Application:** USPTO #19/169,399
- **Claims File:** Listed in `CLAIMS_PRACTICING_FILES.txt`
