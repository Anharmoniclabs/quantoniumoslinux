#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  QuantoniumOS — Replit Deployment Startup Script                    ║
# ║  DeepSeek R1-Distill-Qwen-7B (Q4_K_M) on llama.cpp + Mimosa KV   ║
# ╚══════════════════════════════════════════════════════════════════════╝
set -e

MODEL_DIR="models"
MODEL_FILE="$MODEL_DIR/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
LLAMA_DIR="llama.cpp"
SERVER_BIN="$LLAMA_DIR/build/bin/llama-server"
MIMOSA_TEST_BIN="$LLAMA_DIR/build/bin/llama-mimosa-kv-test"

# ── 1. Clone llama.cpp if not present ─────────────────────────────────
if [ ! -d "$LLAMA_DIR" ]; then
  echo "[setup] Cloning llama.cpp..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
fi

# ── 2. Copy Mimosa KV test into llama.cpp ─────────────────────────────
if [ ! -d "$LLAMA_DIR/examples/mimosa-kv-test" ]; then
  echo "[setup] Installing Mimosa KV test..."
  mkdir -p "$LLAMA_DIR/examples/mimosa-kv-test"
  cp mimosa-kv-test/mimosa_kv_test.cpp "$LLAMA_DIR/examples/mimosa-kv-test/"
  cp mimosa-kv-test/CMakeLists.txt "$LLAMA_DIR/examples/mimosa-kv-test/"

  # Patch the examples CMakeLists to include our test
  if ! grep -q "mimosa-kv-test" "$LLAMA_DIR/examples/CMakeLists.txt"; then
    sed -i '/add_subdirectory(simple-chat)/a\    add_subdirectory(mimosa-kv-test)' \
      "$LLAMA_DIR/examples/CMakeLists.txt"
  fi
fi

# ── 3. Build llama.cpp (server + mimosa test) ─────────────────────────
if [ ! -f "$SERVER_BIN" ] || [ ! -f "$MIMOSA_TEST_BIN" ]; then
  echo "[setup] Building llama.cpp (this takes a few minutes on first run)..."
  cd "$LLAMA_DIR"
  cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_CURL=OFF -DGGML_METAL=OFF -DGGML_CUDA=OFF 2>&1 | tail -3
  cmake --build build -j$(nproc) --target llama-server llama-mimosa-kv-test 2>&1 | tail -5
  cd ..
  echo "[setup] Build complete."
fi

# ── 4. Download model if not present ──────────────────────────────────
if [ ! -f "$MODEL_FILE" ]; then
  echo "[setup] Downloading DeepSeek-R1-Distill-Qwen-7B Q4_K_M (~4.4 GB)..."
  echo "[setup] This is a one-time download."
  mkdir -p "$MODEL_DIR"
  curl -L --progress-bar -o "$MODEL_FILE" \
    "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
  echo "[setup] Model downloaded."
fi

# ── 5. Start the server ───────────────────────────────────────────────
PORT="${PORT:-8080}"
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  QuantoniumOS — R1-Distill-Qwen-7B + Mimosa KV Cache   ║"
echo "║  Server: llama.cpp ($(uname -m), CPU)                   ║"
echo "║  Port: $PORT                                            ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

exec node server.js
