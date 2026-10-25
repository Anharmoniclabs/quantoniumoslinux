#!/usr/bin/env bash
# build.sh — Compile the native inference binary on Replit (no CMake needed)
# Usage: bash build.sh
set -e

CC="${CC:-gcc}"
CXX="${CXX:-g++}"

echo "[build] Compiling C kernel objects..."

# Detect AVX2 support (Replit VMs usually have it)
AVX_FLAGS=""
if $CC -mavx2 -E -x c /dev/null >/dev/null 2>&1; then
  AVX_FLAGS="-mavx2 -mfma"
  echo "[build] AVX2+FMA enabled"
else
  echo "[build] AVX2 not available, using scalar fallback"
fi

CFLAGS="-O3 -std=c99 -DNDEBUG -DRFTMW_ENABLE_ASM=1 -DRFTMW_ENABLE_NATIVE_GEMV_ASM=0 -DRFTMW_ENABLE_SPECTRAL_DOT_ASM=0 $AVX_FLAGS"
CXXFLAGS="-O3 -std=c++17 -DNDEBUG -DRFTMW_ENABLE_ASM=1 -DRFTMW_ENABLE_NATIVE_GEMV_ASM=0 -DRFTMW_ENABLE_SPECTRAL_DOT_ASM=0 $AVX_FLAGS"

INCLUDES="-Isrc/rftmw_native -Ialgorithms/rft/kernels/include -Ialgorithms/rft/kernels/kernel -Ialgorithms/rft/kernels/engines/crypto/include"

mkdir -p build

# C kernel sources
$CC $CFLAGS $INCLUDES -c src/rftmw_native/rftmw_spectral_kernels.c -o build/rftmw_spectral_kernels.o
$CC $CFLAGS $INCLUDES -c algorithms/rft/kernels/kernel/rft_kernel.c -o build/rft_kernel.o
$CC $CFLAGS $INCLUDES -c algorithms/rft/kernels/kernel/rft_kernel_ui.c -o build/rft_kernel_ui.o
$CC $CFLAGS $INCLUDES -c algorithms/rft/kernels/kernel/quantum_symbolic_compression.c -o build/quantum_symbolic_compression.o
$CC $CFLAGS $INCLUDES -c algorithms/rft/kernels/engines/crypto/src/feistel_round48.c -o build/feistel_round48.o
$CC $CFLAGS $INCLUDES -c algorithms/rft/kernels/engines/crypto/src/sha256_portable.c -o build/sha256_portable.o
$CC $CFLAGS $INCLUDES -c algorithms/rft/kernels/engines/crypto/src/rft_sis.c -o build/rft_sis.o

echo "[build] Compiling C++ sources..."

# C++ sources
$CXX $CXXFLAGS $INCLUDES -c src/rftmw_native/rftmw_inference_bridge.cpp -o build/rftmw_inference_bridge.o
$CXX $CXXFLAGS $INCLUDES -c src/rftmw_native/qwen25_native_cli.cpp -o build/qwen25_native_cli.o

echo "[build] Linking..."

$CXX $CXXFLAGS -o build/rftmw_qwen25_native \
  build/qwen25_native_cli.o \
  build/rftmw_inference_bridge.o \
  build/rftmw_spectral_kernels.o \
  build/rft_kernel.o \
  build/rft_kernel_ui.o \
  build/quantum_symbolic_compression.o \
  build/feistel_round48.o \
  build/sha256_portable.o \
  build/rft_sis.o \
  -lm -lpthread

echo "[build] Done: build/rftmw_qwen25_native"
ls -lh build/rftmw_qwen25_native
