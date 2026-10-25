#!/usr/bin/env bash
# Build rftmwd/rftmwctl in a Linux tree outside /mnt/c (fast, case-sensitive).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PATH="$HOME/.local/qos-tools/root/usr/bin:$PATH"
B="$HOME/.cache/qos-build/rftmwd"
cmake -S "$ROOT/linux/rftmwd" -B "$B" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$B" -j"$(nproc)" 2>&1 | grep -v "^\[" || true
ls -la "$B/rftmwd" "$B/rftmwctl"
