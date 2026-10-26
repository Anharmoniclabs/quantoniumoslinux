#!/usr/bin/env bash
# Build the RFTMW kernel benchmark against the baseline and current sources, run
# both, and check the current outputs match the baseline.
#
#   linux/bench/run_bench.sh [quick] [march]      (march default: native)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
NATIVE="$SRC/runtime/src/rftmw_native"
MODE="${1:-full}"; MARCH="${2:-native}"
B="${QOS_BENCH_DIR:-$HOME/.cache/qos-bench}"
mkdir -p "$B"
NASM="${NASM:-$(command -v nasm || echo "$HOME/.local/qos-tools/root/usr/bin/nasm")}"

build() {  # name include_dir
    local name=$1 inc=$2
    "$NASM" -f elf64 "$inc/rftmw_sgemv_f32.asm" -o "$B/$name-sgemv.o"
    "$NASM" -f elf64 "$inc/rftmw_spectral_dot.asm" -o "$B/$name-spec.o"
    g++ -std=c++17 -O3 -march="$MARCH" -DNDEBUG -fopenmp -Wno-unknown-pragmas -I"$inc" \
        "$HERE/bench_rftmw.cpp" "$B/$name-sgemv.o" "$B/$name-spec.o" -o "$B/bench-$name" -pthread
}
build baseline "$HERE/baseline"
build current "$NATIVE"

echo "=== baseline"; "$B/bench-baseline" "$B/baseline.bin" "$MODE" | tee "$B/baseline.txt"
echo "=== current";  "$B/bench-current"  "$B/current.bin"  "$MODE" | tee "$B/current.txt"
echo "=== equivalence (current vs baseline)"
python3 "$HERE/compare.py" "$B/baseline.bin" "$B/current.bin" "${QOS_BENCH_TOL:-1e-4}"
