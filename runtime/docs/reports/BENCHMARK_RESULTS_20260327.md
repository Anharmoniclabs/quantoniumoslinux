# Benchmark Results 2026-03-27

Source artifact: [benchmarks/results/bench_all_actual.json](C:\Users\mkeln\quantoniumos\benchmarks\results\bench_all_actual.json)

Environment:
- Python `3.12.9`
- Native module `rftmw_native` available
- Installed comparator packages: `qiskit`, `qiskit-aer`, `cirq`, `pyfftw`, `mkl_fft`, `zstandard`, `brotli`, `lz4`, `cryptography`, `pynacl`, `liboqs-python`, `librosa`, `sounddevice`, `pydub`
- Local `liboqs` build installed under `third_party/_oqs`

## Executive Summary

- Class A now includes actual Qiskit and Cirq timings alongside the quantoniumos symbolic path.
- Class B includes actual FFTW and MKL comparisons; Φ-RFT remains orders of magnitude slower than FFT-family baselines at tested sizes.
- Class C now reports actual RFTMW measurements for every dataset using a benchmark-side chunked mode (`actual-chunked-1024`) instead of simulated entries.
- Class D now includes actual AES-GCM, ChaCha20-Poly1305, Kyber-512, and Dilithium-2 metrics in the JSON artifact.
- Class E includes actual library timings and an actual Φ-RFT timing on a `64`-sample frame.

## Class A: Quantum Symbolic Simulation

Classical full-state simulators:

| Qubits | Qiskit (ms) | Cirq (ms) |
|---|---:|---:|
| 4 | 0.45 | 1.53 |
| 6 | 0.35 | 1.23 |
| 8 | 0.48 | 1.93 |
| 10 | 0.63 | 1.76 |
| 12 | 0.89 | 2.48 |
| 14 | 3.54 | 2.74 |
| 16 | 18.28 | 3.82 |
| 18 | 75.36 | 7.81 |
| 20 | 441.36 | 25.70 |
| 22 | 1943.04 | 107.53 |
| 24 | 8576.94 | 452.32 |

quantoniumos symbolic path:

| Symbolic qubits | Time (ms) | Compression size | Entropy |
|---|---:|---:|---:|
| 10 | 0.01 | 64 | 0.001389 |
| 100 | 0.01 | 64 | 0.008813 |
| 1,000 | 0.05 | 64 | 0.009751 |
| 10,000 | 0.52 | 64 | 0.009771 |
| 100,000 | 5.00 | 64 | 0.009818 |
| 1,000,000 | 58.00 | 64 | 0.009867 |
| 10,000,000 | 608.00 | 64 | 0.009857 |

Notes:
- Qiskit/Cirq are exact amplitude simulators.
- QSC is a symbolic/compressed classical representation and is not equivalent to a full `2^n` statevector simulation.

## Class B: Transform and DSP

| Size | NumPy FFT (us) | SciPy FFT (us) | FFTW (us) | MKL (us) | Native Phi-RFT (us) | Ratio vs NumPy | H3 Hybrid (us) | H3 BPP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 5.83 | 11.67 | 1.60 | 7.40 | 102390.30 | 17552.62x | 478.73 | 0.492 |
| 128 | 4.67 | 8.87 | 2.43 | 3.27 | 850362.30 | 182220.43x | 464.63 | 0.620 |

Energy compaction from FFT top-10% coefficients:

| Signal | Energy in top 10% | Interpretation |
|---|---:|---|
| random | 31.0% | spread spectrum |
| sine | 100.0% | highly compressible |
| ascii | 100.0% | highly compressible |
| sparse | 81.1% | moderate structure |
| chirp | 99.7% | highly compressible |

## Class C: Compression

RFTMW entries below are actual benchmark measurements in chunked mode with `1024`-byte blocks.

| Dataset | Size | gzip | LZMA | zstd | brotli | LZ4 | RFTMW |
|---|---:|---:|---:|---:|---:|---:|---:|
| code | 53,000 | 101.15x | 142.47x | 204.63x | 232.46x | 93.64x | 3.85x |
| text | 57,400 | 117.86x | 161.24x | 268.22x | 329.89x | 106.69x | 4.40x |
| json | 93,789 | 7.48x | 13.22x | 9.11x | 10.76x | 3.93x | 4.20x |
| random | 100,000 | 1.00x | 1.00x | 1.00x | 1.00x | 1.00x | 1.07x |
| pattern | 100,000 | 564.97x | 641.03x | 3571.43x | 5263.16x | 224.22x | 22.68x |

Text throughput:

| Codec | Compress | Decompress |
|---|---:|---:|
| gzip | 243.3 MB/s | 1227.4 MB/s |
| LZMA | 16.1 MB/s | 257.5 MB/s |
| zstd | 846.1 MB/s | 1948.1 MB/s |
| brotli | 102.2 MB/s | 593.1 MB/s |
| LZ4 | 6015.5 MB/s | 6082.3 MB/s |

RFTMW notes:
- `code`, `text`, `json`, `random`, and `pattern` are all tagged `actual-chunked-1024` in the JSON artifact.
- This chunked mode was added because one-shot `rft_compress` attempted `N x N` allocations on these buffers and failed at `20` to `75` GiB.
- Shannon side-suite still timed out after the aggregate runner's `30s` cap.

## Class D: Cryptography and Post-Quantum

Hash timings:

| Algorithm | Time (us) | Throughput | Output |
|---|---:|---:|---:|
| SHA-256 | 1.45 | 708.3 MB/s | 32 B |
| SHA3-256 | 3.68 | 278.3 MB/s | 32 B |
| BLAKE2b | 2.80 | 366.1 MB/s | 64 B |
| RFT-SIS | 4101.27 | 0.25 MB/s | 32 B |

AEAD / cipher timings:

| Cipher | Encrypt (us) | Decrypt (us) | Status |
|---|---:|---:|---|
| AES-256-GCM | 1.45 | 1.39 | actual |
| ChaCha20-Poly1305 | 5.32 | 6.10 | actual |
| RFT-Feistel-48 | 56061.67 | 51707.65 | Python fallback |

Post-quantum metrics:

| Algorithm | Keygen (us) | Operation 1 | Operation 2 | Public key | Ciphertext / Signature |
|---|---:|---:|---:|---:|---:|
| Kyber-512 | 220.81 | Encap 132.85 | Decap 56.76 | 800 B | 768 B |
| Dilithium-2 | 166.96 | Sign 396.78 | Verify 95.67 | 1312 B | 2420 B |

Avalanche:

| Algorithm | Avalanche |
|---|---:|
| SHA-256 | 49.74% |
| SHA3-256 | 50.13% |
| BLAKE2b | 50.20% |
| RFT-SIS | 50.06% |

Notes:
- `RFT-SIS` and `RFT-Feistel-48` are still benchmarked through Python fallback paths in this environment.
- Kyber and Dilithium are actual `liboqs` measurements and are now present in the JSON artifact under `classes.D.data.pq`.

## Class E: Audio and DAW

| Algorithm | Time (us) | Latency (ms) | Note |
|---|---:|---:|---|
| NumPy FFT | 537.00 | 0.537 | actual |
| SciPy STFT | 1385.14 | 1.385 | actual |
| librosa MelSpec | 62813.78 | 62.814 | actual |
| SciPy Butterworth | 329.92 | 0.330 | actual |
| Phi-RFT Transform | 121593.48 | 121.593 | actual `64`-sample frame |

Audio I/O snapshot:
- Buffer size: `256`
- Sample rate: `44100`
- Buffer latency: `5.80 ms`
- Input latency: `90.00 ms`
- Output latency: `90.00 ms`

## Warnings and Limits

- `liboqs` runtime warning remains: local `liboqs` is `0.15.0` while `liboqs-python` is `0.14.1`.
- `experiments/hypothesis_testing/hybrid_mca_fixes.py:108` still emits a `ComplexWarning`.
- `pydub` warns that `ffmpeg`/`avconv` is not installed.
- Class C RFTMW numbers are actual, but they are measured in benchmark-side chunked mode rather than one-shot whole-buffer mode because the current one-shot path is not memory-safe on these dataset sizes.
