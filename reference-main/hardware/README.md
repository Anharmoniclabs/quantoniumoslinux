# /hardware/ — RTL CONCEPT DESIGN ONLY

> ⚠️ **NO FABRICATED SILICON. NO DEPLOYED FPGA BITSTREAM.**  
> All results in this directory are from RTL simulation and EDA synthesis
> tools (Yosys, WebFPGA). No physical chip or programmed FPGA board exists.
> Performance figures are synthesis estimates, not measured hardware.

## ARCHITECTURAL FEASIBILITY STUDY

Mandatory labels:
- **No silicon fabricated**
- **No FPGA bitstream deployed to physical hardware**
- **Performance based on RTL simulation / synthesis estimation only**
- **Not a claim of optimality**

This tree contains RTL/TL-Verilog/FPGA builds, testbenches, synthesis
reports, and visualization tooling for the RFTPU concept accelerator.

### Verified Configuration

The `fpga_top.json` file defines the 16-mode configuration used for
WebFPGA synthesis. This configuration reflects the **Verified Scientific
Findings** (Dec 2025):

- **Mode 0 (Golden):** Verified Class B Transform.
- **Mode 6 (Cascade):** Verified Class B Compression Winner (H3).
- **Mode 12 (SIS Hash):** Verified Class D Crypto — **research only, NOT for production.**
- **Mode 14 (Quantum-Inspired Symbolic):** Classical compression engine for
  amplitude vectors; operates on standard-sized NumPy arrays. Not a quantum
  circuit simulator. No qubits, no quantum gates.

### Subsections

- `hardware/rtl/` — RTL sources and architecture definitions
- `hardware/simulation/` — testbenches and simulation harnesses
- `hardware/architecture_docs/` — design docs, reports, and figures
