#!/usr/bin/env bash
# QuantoniumOS build: compiles the whole stack in a QuantoniumOS tree.
#
#   linux/build.sh [TREE]        TREE defaults to the checkout containing this script
#
# Layout expected in TREE:  runtime/  (QuantoniumOS sources)   linux/  (this OS layer)
# Produces, inside TREE:
#   build/rftmwd/{rftmwd,rftmwctl}                    middleware daemon + client
#   runtime/src/rftmw_native/build/                   rftmw_native python module + native CLIs
#   runtime/.venv/                                    python engines' environment
#   runtime/desktop-shell/src-tauri/target/release/quantoniumos-desktop
#
# Environment:
#   QOS_MARCH=native|x86-64-v3   CPU baseline for C kernels (ISO builds use x86-64-v3)
#   QOS_SKIP_DESKTOP=1           skip the Tauri shell (headless / CI builds)
set -euo pipefail

TREE="$(cd "${1:-$(dirname "$0")/..}" && pwd)"
MARCH="${QOS_MARCH:-native}"
JOBS="$(nproc)"
step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

[ -d "$TREE/runtime/src/rftmw_native" ] || { echo "not a QuantoniumOS tree: $TREE" >&2; exit 1; }

step "rftmwd middleware daemon"
cmake -S "$TREE/linux/rftmwd" -B "$TREE/build/rftmwd" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$TREE/build/rftmwd" -j "$JOBS"

step "Python environment for the RFTMW engines"
if [ ! -x "$TREE/runtime/.venv/bin/python" ]; then
    # System numpy/scipy/pybind11 come from pacman; the venv layers the rest on top.
    python -m venv --system-site-packages "$TREE/runtime/.venv"
fi
"$TREE/runtime/.venv/bin/pip" install --quiet --disable-pip-version-check -r "$TREE/runtime/requirements.txt"

step "rftmw_native: NASM + C kernels, C++ engine, pybind11 module, native CLIs (-march=$MARCH)"
cmake -S "$TREE/runtime/src/rftmw_native" -B "$TREE/runtime/src/rftmw_native/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DRFTMW_MARCH="$MARCH" \
    -DPython3_EXECUTABLE="$TREE/runtime/.venv/bin/python" \
    -DRFTMW_ENABLE_ASM=ON -DRFTMW_ENABLE_NATIVE_GEMV_ASM=ON -DENABLE_LTO=ON
cmake --build "$TREE/runtime/src/rftmw_native/build" -j "$JOBS"
"$TREE/runtime/.venv/bin/python" -c "import sys; sys.path.insert(0, '$TREE/runtime/src/rftmw_native/build'); import rftmw_native; print('rftmw_native OK:', rftmw_native.__file__)"

step "Prebuilt canonical RFT bases (installed to /usr/share/quantoniumos/rftmw-basis)"
# Each size costs O(N^3) to build (about 1-2 s at N=256); shipping them makes the
# first transform of a session as fast as every later one.
rm -rf "$TREE/build/rftmw-basis"
RFTMW_BASIS_DIR="$TREE/build/rftmw-basis" "$TREE/runtime/.venv/bin/python" - "$TREE/runtime/src/rftmw_native/build" <<'EOF'
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
import rftmw_native as r
eng = r.RFTMWEngine(256)
for n in (8, 16, 32, 64, 128, 256):
    eng.forward(np.zeros(n))
EOF
ls "$TREE/build/rftmw-basis"

if [ "${QOS_SKIP_DESKTOP:-0}" != 1 ]; then
    step "QuantoniumOS shell (Tauri)"
    cd "$TREE/runtime/desktop-shell"
    npm ci --no-audit --no-fund
    npm run tauri build -- --no-bundle
    ls -la "$TREE/runtime/desktop-shell/src-tauri/target/release/quantoniumos-desktop"
fi

step "Build complete"
