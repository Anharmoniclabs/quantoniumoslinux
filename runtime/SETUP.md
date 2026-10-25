# Setup and Reproducibility

One-command setup for the RFT library, benchmarks, and theorem test suite.

## Quick Start

```bash
git clone https://github.com/LMMinier/quantoniumos.git
cd quantoniumos
pip install -e ".[benchmark]"
python algorithms/rft/applications/vibration/cwru_benchmark.py
```

Full documented results: algorithms/rft/applications/vibration/CWRU_BENCHMARK_RESULTS.md

## Install Options

| Command | What it installs | Use case |
|---|---|---|
| pip install -e . | Core only | Just the RFT math |
| pip install -e ".[benchmark]" | Core + pyfftw, librosa | Run all benchmarks |
| pip install -e ".[dev]" | Core + pytest, black, flake8 | Development |
| pip install -e ".[dev,benchmark]" | Everything above | Full local dev |
| pip install -r requirements.txt | Minimal flat install | Fastest path |

All dependency groups are defined in pyproject.toml.

## Run All Benchmarks

```bash
python algorithms/rft/applications/vibration/cwru_benchmark.py
python benchmarks/kv_cache_boundary_error.py
python theorem_e_proof_test.py
```

## Run All Theorem Tests

```bash
pytest tests/proofs/ -v
```

## Full Environment Verification

```bash
./verify_setup.sh
./reproduce_results.sh
```

## System Requirements

- Python 3.10, 3.11, or 3.12
- CPU only - no GPU required
- No quantum hardware - all RFT is classical signal processing
- Linux, macOS, or Windows (WSL recommended)

## Automated CI

Every push to main that touches algorithms/rft or src triggers:
- ci.yml - full test suite across Python 3.10/3.11/3.12
- shannon_tests.yml - entropy and information-theoretic tests
- security-scan.yml - dependency vulnerability scan

## Citation

```
@misc{rft2025,
  title  = {Resonant Fourier Transform: Golden-Ratio Multi-Carrier Wave Encoding},
  author = {Minier, Luis M.},
  year   = {2025},
  note   = {USPTO Patent Application 19/169,399}
}
```

Zenodo: https://zenodo.org/records/15163648

Origin story: ORIGIN.md
