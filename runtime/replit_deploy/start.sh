#!/usr/bin/env bash
# start.sh — Download model (if missing), build native binary, start server
set -e

MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/qwen2.5-coder-3b-instruct-q4_k_m.gguf"

# 1. Download model if not present
if [ ! -f "$MODEL_FILE" ]; then
  echo "[start] Downloading Qwen2.5-Coder-3B Q4_K_M GGUF (~2 GB)..."
  mkdir -p "$MODEL_DIR"
  curl -L -o "$MODEL_FILE" \
    "https://huggingface.co/Qwen/Qwen2.5-Coder-3B-Instruct-GGUF/resolve/main/qwen2.5-coder-3b-instruct-q4_k_m.gguf"
  echo "[start] Model downloaded."
fi

# 2. Build if binary not present
if [ ! -f "build/rftmw_qwen25_native" ]; then
  echo "[start] Building native binary..."
  bash build.sh
fi

# 3. Set env and start the Node bridge
export NATIVE_BIN="build/rftmw_qwen25_native"
export GGUF_PATH="$MODEL_FILE"
export RFTMW_PORT="${PORT:-8000}"
export RFTMW_MIMOSA_CACHE="${RFTMW_MIMOSA_CACHE:-512}"

echo "[start] Starting server on port $RFTMW_PORT..."
exec node mobile/server/rftmw_native_serve.js
