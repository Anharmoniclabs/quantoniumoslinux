#!/usr/bin/env bash
# Build all of rftmw_native (pybind module, Qwen CLIs, tests) on Linux and run the
# native round-trip test and a Python smoke test of the module. Runs inside the Arch
# chroot (see wsl_arch_run.sh) or any Arch/Linux box with the build deps installed.
#   wsl_arch_run.sh 'bash /src/linux/tests/native_build_check.sh'
set -euo pipefail
S="${1:-/src/runtime/src/rftmw_native}"
B="${QOS_NATIVE_BUILD:-/var/tmp/qos-native-check}"
rm -rf "$B"
cmake -S "$S" -B "$B" -G Ninja -DCMAKE_BUILD_TYPE=Release -DRFTMW_MARCH=x86-64-v3 -DENABLE_LTO=ON \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" > "$B.cfg.log" 2>&1 || { tail -30 "$B.cfg.log"; exit 1; }
grep -E "OpenMP|ASM kernels" "$B.cfg.log"
cmake --build "$B" --parallel > "$B.build.log" 2>&1 || { grep -E "error|FAILED" "$B.build.log" | head -30; exit 1; }
grep -c "warning" "$B.build.log" | sed 's/^/compiler warnings: /'
for f in "$B"/rftmw* "$B"/test_*; do [ -f "$f" ] && echo "built: ${f##*/}"; done

echo "--- test_spectral_roundtrip"
"$B/test_spectral_roundtrip" | tail -8

echo "--- python module"
RFTMW_BASIS_DIR="$(mktemp -d)" PYTHONPATH="$B" python - <<'EOF'
import time
import numpy as np
import rftmw_native as r

eng = r.RFTMWEngine(256)
rng = np.random.default_rng(0)
for n in (64, 256, 2048):
    x = rng.standard_normal(n)
    t0 = time.perf_counter(); X = eng.forward(x); cold = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        X = eng.forward(x)
    warm = (time.perf_counter() - t0) / 20
    y = eng.inverse(X)
    err = float(np.max(np.abs(y - x))) if n <= 256 else float("nan")
    print(f"N={n:5d} cold {cold*1e3:8.2f} ms  warm {warm*1e3:8.3f} ms  roundtrip_err {err:.2e}")
    if n <= 256:
        assert err < 1e-10, err
print("python module OK")
EOF
