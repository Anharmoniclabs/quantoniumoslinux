#!/usr/bin/env bash
# Assemble the RFTMW kernels as elf64 and run the cross-ABI test (System V side).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
NASM="${NASM:-$(command -v nasm || echo "$HOME/.local/qos-tools/root/usr/bin/nasm")}"
SRC="$ROOT/runtime/src/rftmw_native"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
"$NASM" -f elf64 "$SRC/rftmw_sgemv_f32.asm" -o "$OUT/sgemv.o"
"$NASM" -f elf64 "$SRC/rftmw_spectral_dot.asm" -o "$OUT/sdot.o"
gcc -O2 -mavx2 -mfma "$ROOT/linux/tests/test_asm_abi.c" "$OUT/sgemv.o" "$OUT/sdot.o" -lm -z noexecstack -o "$OUT/test_asm_abi"
"$OUT/test_asm_abi"
