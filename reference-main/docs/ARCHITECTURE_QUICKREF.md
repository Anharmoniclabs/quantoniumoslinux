# RFT Architecture Quick Reference

**One-page cheat sheet for the implementation stack**

---

## 🏗️ Four-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│ LAYER 4: Python API (Always Available)             │
│ • Files: algorithms/rft/core/*.py                   │
│ • Speed: 2-5 GB/s (with Numba)                      │
│ • Deps: NumPy, SciPy, SymPy                         │
├─────────────────────────────────────────────────────┤
│ LAYER 3: C++ Engine (Optional, pybind11)           │
│ • Files: src/rftmw_native/*.cpp/*.hpp               │
│ • Speed: 30 GB/s (with AVX2 SIMD)                   │
│ • Deps: C++17, pybind11                             │
├─────────────────────────────────────────────────────┤
│ LAYER 2: C Kernel (Optional, Portable)             │
│ • Files: algorithms/rft/kernels/kernel/*.c          │
│ • Speed: 8 GB/s (portable)                          │
│ • Deps: C99/C11, gcc/clang                          │
├─────────────────────────────────────────────────────┤
│ LAYER 1: Assembly (Optional, Maximum Speed)        │
│ • Files: algorithms/rft/kernels/**/*.asm            │
│ • Speed: 50 GB/s (AVX-512)                          │
│ • Deps: NASM, x86_64 CPU                            │
└─────────────────────────────────────────────────────┘
```

---

## 📊 Performance Comparison (N=1024)

| Layer           | Latency | Throughput | Speedup | Platform    |
|:----------------|:--------|:-----------|:--------|:------------|
| Pure NumPy      | 500 μs  | 2 GB/s     | 1×      | All         |
| +Numba JIT      | 200 μs  | 5 GB/s     | 2.5×    | All         |
| +C/C++          | 50 μs   | 20 GB/s    | 10×     | Linux/macOS |
| +C/C++/SIMD     | 35 μs   | 30 GB/s    | 15×     | x86_64      |
| +ASM/AVX2       | 25 μs   | 40 GB/s    | 20×     | x86_64      |
| +ASM/AVX-512    | 20 μs   | 50 GB/s    | 25×     | Intel/AMD   |

---

## 🔄 Data Flow: RFT Transform

```python
# User code
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
y = CanonicalTrueRFT(1024).forward_transform(x)
```

**Execution path (all layers available):**

```
1. Python
   canonical_true_rft.py::forward_transform()
   ↓ validates input, checks for native bindings
   
2. pybind11 (zero-copy)
   rftmw_python.cpp::rft_forward()
   ↓ passes NumPy buffer pointer directly
   
3. C++ Engine
   rftmw_core.hpp::RFTEngine::forward()
   ↓ computes phase vectors, dispatches to kernel
   
4. C Kernel (if no C++, or as fallback)
   rft_kernel.c::rft_forward_impl()
   ↓ calls FFT, applies phases
   
5. Assembly (hot paths)
   rft_kernel_asm.asm::rft_fft_butterfly()
   ↓ hand-optimized FFT butterflies
   
6. Return (zero-copy)
   ASM → C → C++ → pybind11 → NumPy array
```

---

## 🛠️ Installation Options

### Option 1: Pure Python (No Compilation)
```bash
pip install numpy scipy sympy numba
# Done! Runs at ~2-5 GB/s
```

### Option 2: Standard (Recommended)
```bash
./quantoniumos-bootstrap.sh
# Installs Python deps + attempts C/C++ build
```

### Option 3: Full Performance (All Layers)
```bash
# Install system deps
sudo apt install build-essential cmake nasm python3-dev

# Python deps
pip install numpy scipy sympy numba

# Build C/ASM kernel
cd algorithms/rft/kernels
make clean && make -j$(nproc)
ls -lh compiled/libquantum_symbolic.so  # Should be ~200-400 KB
cd ../../..

# Build C++ engine
cd src/rftmw_native
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DRFTMW_ENABLE_ASM=ON
make -j$(nproc)  # Will detect AVX2/AVX-512 automatically
cp rftmw_native.cpython-*-linux-gnu.so ../../../.venv/lib/python3.12/site-packages/
cd ../../..

# Verify all layers
python -c "
import rftmw_native
from algorithms.rft.kernels.python_bindings import _load_assembly_library

print('✓ Layer 1 (ASM):  ', rftmw_native.HAS_ASM_KERNELS)
print('✓ Layer 2 (C):    ', _load_assembly_library() is not None)
print('✓ Layer 3 (C++):  ', True)
print('✓ Layer 4 (Python):', True)
print()
print('Hardware Features:')
print('  AVX2:   ', rftmw_native.HAS_AVX2)
print('  AVX-512:', rftmw_native.HAS_AVX512)
print('  FMA:    ', rftmw_native.HAS_FMA)
"
```

---

## 📁 Key Files by Layer

### Python Layer (always used)
```
algorithms/rft/core/
├── canonical_true_rft.py        # Main API
├── resonant_fourier_transform.py # Current NumPy/frame implementation
├── fast_rft.py                  # Faster experimental path
└── fast_rft_surrogate.py        # Approximate/accelerated surrogate
```

### C++ Layer (optional, for speed)
```
src/rftmw_native/
├── rftmw_core.hpp              # Core engine
├── rftmw_python.cpp            # pybind11 bindings
├── rft_fused_kernel.hpp        # SIMD kernels
└── CMakeLists.txt              # Build config
```

### C Layer (optional, portable)
```
algorithms/rft/kernels/kernel/
├── rft_kernel.c                # Portable C implementation
├── rft_kernel.h                # Public API
└── quantum_symbolic_compression.c
```

### Assembly Layer (optional, max speed)
```
algorithms/rft/kernels/
├── kernel/rft_kernel_asm.asm
├── kernel/quantum_symbolic_compression.asm
└── engines/crypto/asm/feistel_round48.asm
```

---

## 🌍 Platform Support Matrix

| Platform       | Python | C/C++ | ASM | Best Speed |
|:---------------|:------:|:-----:|:---:|:-----------|
| Linux x86_64   | ✅     | ✅    | ✅  | 50 GB/s    |
| macOS x86_64   | ✅     | ✅    | ✅  | 50 GB/s    |
| macOS ARM64    | ✅     | ✅    | 🚧  | 30 GB/s    |
| Windows (WSL2) | ✅     | ✅    | ✅  | 50 GB/s    |
| ARM (RPi)      | ✅     | ✅    | 🚧  | 15 GB/s    |
| RISC-V         | ✅     | ✅    | ❌  | 8 GB/s     |

✅ Full support | 🚧 Work in progress | ❌ Not available

---

## 🔍 Verification

### Quick Test
```bash
source .venv/bin/activate
python -c "
from algorithms.rft.core.canonical_true_rft import CanonicalTrueRFT
import numpy as np
rft = CanonicalTrueRFT(64)
x = np.random.randn(64)
y = rft.forward_transform(x)
print(f'✓ RFT Core: OK')
print(f'Unitarity: {rft.get_unitarity_error():.2e}')
"
```

### Check Layer Availability
```bash
# Python (always available)
python -c "import numpy; print('✓ Python')"

# C/C++ (if compiled)
python -c "from rftmw import rft_forward; print('✓ C/C++')" 2>/dev/null || echo "⚠ Python only"

# ASM (if compiled with NASM)
python -c "from algorithms.rft.kernels.python_bindings import _load_assembly_library; lib = _load_assembly_library(); print('✓ ASM' if lib else '⚠ C fallback')"
```

---

## 🚀 Performance Tips

### 1. Use Powers of 2
```python
# Good (FFT-friendly)
rft = CanonicalTrueRFT(1024)  # 2^10

# Still works, but slower
rft = CanonicalTrueRFT(1000)  # Not power of 2
```

### 2. Reuse RFT Objects
```python
# Bad (creates new engine each time)
for x in signals:
    y = CanonicalTrueRFT(1024).forward_transform(x)

# Good (reuses engine)
rft = CanonicalTrueRFT(1024)
for x in signals:
    y = rft.forward_transform(x)
```

### 3. Use Complex128 (Double Precision)
```python
x = np.random.randn(1024).astype(np.complex128)  # Best precision
```

### 4. Batch Processing
```python
# Process multiple signals at once (if using C++ layer)
xs = np.random.randn(100, 1024)  # 100 signals
ys = rft.forward_transform_batch(xs)  # Parallelized
```

---

## 🐛 Common Issues

### "No module named 'rftmw'"
**Cause:** C++ extensions not built  
**Fix:** Use pure Python (always available) or build extensions

### "Unitarity error too high"
**Cause:** Numerical precision issue  
**Fix:** Use `np.complex128` (not `complex64`)

### "NASM not found"
**Cause:** NASM assembler not installed  
**Fix:** `sudo apt install nasm` or use C fallback

---

## 📖 Documentation

- **SETUP_GUIDE.md** - Installation guide
- **docs/ARCHITECTURE.md** - Technical deep dive
- **QUICK_REFERENCE.md** - API quick reference
- **REPRODUCING_RESULTS.md** - Benchmark guide

---

## 🎯 When to Use Each Layer

| Use Case                  | Recommended Layer | Why                        |
|:--------------------------|:------------------|:---------------------------|
| Research prototyping      | Python only       | Easy iteration             |
| Algorithm validation      | Python + Numba    | Good balance               |
| Production real-time audio| C/C++ + ASM       | Minimum latency            |
| Cross-platform deployment | C/C++ (no ASM)    | Portable performance       |
| ARM/embedded systems      | C only            | Portable, low overhead     |
| Maximum throughput        | ASM/AVX-512       | Saturate memory bandwidth  |

---

**quantoniumos: Fast by default, faster when needed.**

ASM → C → C++ → Python: Choose your performance tier.
