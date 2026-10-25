#!/usr/bin/env bash
# qos-run / qos-status against a live rftmwd (systemd user manager required).
set -uo pipefail
L="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${1:-$HOME/.cache/qos-build/rftmwd}"
T="$(mktemp -d -p "$HOME/.cache")"
export RFTMW_SOCKET="$T/s" PATH="$BIN:$L/bin:$PATH"
systemd-run --user --quiet --unit=rftmwd-session-test -p Type=notify --setenv=RFTMW_SOCKET="$RFTMW_SOCKET" "$BIN/rftmwd" --slice rftmwtest
trap 'systemctl --user stop rftmwd-session-test rftmwtest.slice 2>/dev/null; rm -rf "$T"' EXIT
fail=0
qos-run --name tool-test --mem-high 256 -- sleep 20
sleep 1  # systemd-run registers the scope just after rftmwd forks
rftmwctl procs | grep -q '"name":"tool-test","pid":[0-9]*,"running":true.*"memory_high":268435456' && echo "PASS qos-run (scope + MemoryHigh)" || { echo "FAIL qos-run"; fail=1; }
out="$(qos-status)"; echo "$out"
echo "$out" | jq -e '.text | startswith("RFTMW ")' >/dev/null && echo "$out" | jq -e '.tooltip | contains("tool-test")' >/dev/null \
  && echo "PASS qos-status" || { echo "FAIL qos-status"; fail=1; }
systemctl --user stop rftmwd-session-test
# Broker down: qos-run must still run the program (fallback exec).
qos-run -- sh -c 'echo fallback-ran' | grep -q fallback-ran && echo "PASS qos-run fallback" || { echo "FAIL qos-run fallback"; fail=1; }
qos-status | jq -e '.class == "offline"' >/dev/null && echo "PASS qos-status offline" || { echo "FAIL qos-status offline"; fail=1; }
exit $fail
