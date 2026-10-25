#!/usr/bin/env bash
# Start the RFTMW inference server + Expo dev server together.
# Usage: ./start.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Detect Codespace forwarded URLs ──
if [[ -n "${CODESPACE_NAME:-}" ]]; then
  FWD_DOMAIN="${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}"
  SERVER_URL="https://${CODESPACE_NAME}-8000.${FWD_DOMAIN}"
  APP_URL="https://${CODESPACE_NAME}-8081.${FWD_DOMAIN}"
  export EXPO_PUBLIC_DOMAIN="${CODESPACE_NAME}-8000.${FWD_DOMAIN}"
else
  SERVER_URL="http://localhost:8000"
  APP_URL="http://localhost:8081"
fi

cleanup() {
  echo ""
  echo "[start] Shutting down..."
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null
  wait 2>/dev/null
  echo "[start] Done."
}
trap cleanup EXIT INT TERM

# ── Kill any leftover server/expo from previous runs ──
pkill -f "rftmw_serve.py"  2>/dev/null || true
pkill -f "expo.*start"     2>/dev/null || true
sleep 1

# ── 1. RFTMW inference server (port 8000) ──
# RFTMW_PRELOAD_FP32=1 materializes weights as dense FP32 matrices at startup
# (~12 GB RAM) so per-token matmul hits BLAS directly instead of re-dequantizing
# every call. Measured ~3.75x speedup on an i7-6700 with 64 GB RAM.
export RFTMW_PRELOAD_FP32="${RFTMW_PRELOAD_FP32:-1}"
echo "[start] Starting RFTMW inference server (PRELOAD_FP32=$RFTMW_PRELOAD_FP32)..."
cd "$PROJECT_ROOT"
python3 mobile/server/rftmw_serve.py &
SERVER_PID=$!

# Wait for the server to come up
echo "[start] Waiting for server on port 8000..."
for i in $(seq 1 120); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "[start] Server ready."
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[start] ERROR: Server process died." >&2
    exit 1
  fi
  sleep 1
done

# ── 2. Install deps if needed ──
if [[ ! -d "$SCRIPT_DIR/node_modules" ]]; then
  echo "[start] Installing dependencies..."
  cd "$SCRIPT_DIR"
  pnpm install
fi

# ── 3. Expo dev server (port 8081) ──
echo ""
echo "========================================"
echo "  QU.AI Mobile — Running"
echo "========================================"
echo "  Server : $SERVER_URL"
echo "  App    : $APP_URL"
echo ""
echo "  Open the App URL on your phone."
if [[ -n "${CODESPACE_NAME:-}" ]]; then
  echo "  Make sure ports 8000 & 8081 are Public"
  echo "  in the VS Code Ports tab."
fi
echo "========================================"
echo ""

echo "[start] Starting Expo dev server (interactive)..."
cd "$SCRIPT_DIR"
exec npx expo start --port 8081 --host lan
