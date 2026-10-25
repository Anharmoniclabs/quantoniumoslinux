#!/usr/bin/env bash
# End-to-end test of rftmwd. Uses a systemd user unit with Delegate=yes when a user
# manager is available (as on QuantoniumOS), otherwise a plain background process.
set -uo pipefail
BIN="${1:-$HOME/.cache/qos-build/rftmwd}"
T="$(mktemp -d -p "$HOME/.cache")"  # not /tmp: tmpfs pages cannot be evicted
export RFTMW_SOCKET="$T/rftmwd.sock"
CTL="$BIN/rftmwctl"
pass=0 fail=0
check() { local name="$1" expect="$2"; shift 2; local out; out="$("$@" 2>&1)"
  if grep -q -- "$expect" <<<"$out"; then echo "PASS $name"; pass=$((pass+1)); else echo "FAIL $name: $out"; fail=$((fail+1)); fi; }

if systemctl --user is-system-running >/dev/null 2>&1 || [ "$(systemctl --user is-system-running 2>/dev/null)" = degraded ]; then
  systemd-run --user --quiet --unit=rftmwd-test -p Type=notify -p Restart=on-failure -p RestartSec=1 \
    --setenv=RFTMW_SOCKET="$RFTMW_SOCKET" "$BIN/rftmwd" --pressure-idle-s 0 --slice rftmwtest
  cleanup() { systemctl --user stop rftmwd-test rftmwtest.slice 2>/dev/null; systemctl --user reset-failed rftmwd-test 2>/dev/null; rm -rf "$T"; }
  MODE=systemd
else
  "$BIN/rftmwd" --pressure-idle-s 0 2>"$T/log" & DPID=$!
  cleanup() { kill $DPID 2>/dev/null; wait $DPID 2>/dev/null; rm -rf "$T"; }
  MODE=plain
fi
trap cleanup EXIT
for _ in $(seq 50); do [ -S "$RFTMW_SOCKET" ] && break; sleep 0.1; done
echo "mode: $MODE"

check ping          '"pong":true'                 "$CTL" ping
check stats         '"mem_total":'                "$CTL" stats
check rft-selftest  '"roundtrip_max_abs_err":'    "$CTL" rft.selftest 64
"$CTL" rft.selftest 64; echo
check rft-forward   '"coeffs":\[\['               "$CTL" rft.forward 1 0 -1 0 1 0 -1 0
check bad-command   '"ok":false'                  "$CTL" nosuch

# --- memory broker -------------------------------------------------------
head -c $((128<<20)) /dev/urandom > "$T/weights.gguf"; printf 'GGUF' | dd of="$T/weights.gguf" conv=notrunc status=none
check pack-open     '"format":"gguf"'             "$CTL" pack.open "$T/weights.gguf"
check pack-dedupe   '"id":1,'                     "$CTL" pack.open "$T/weights.gguf"
"$CTL" pack.evict 1 >/dev/null
r0=$("$CTL" pack.list | grep -o '"resident_bytes":[0-9]*' | cut -d: -f2)
"$CTL" pack.prefetch 1 >/dev/null; sleep 1
r1=$("$CTL" pack.list | grep -o '"resident_bytes":[0-9]*' | cut -d: -f2)
"$CTL" pack.evict 1 >/dev/null
r2=$("$CTL" pack.list | grep -o '"resident_bytes":[0-9]*' | cut -d: -f2)
echo "residency MiB: after-evict=$((r0>>20)) after-prefetch=$((r1>>20)) after-evict=$((r2>>20))"
if [ "$r1" -gt $((100<<20)) ] && [ "$r2" -lt $((r1/4)) ]; then echo "PASS pack-prefetch-evict"; pass=$((pass+1)); else echo "FAIL pack-prefetch-evict"; fail=$((fail+1)); fi
"$CTL" pack.prefetch 1 >/dev/null; sleep 1
"$CTL" pack.budget 32 >/dev/null
r3=$("$CTL" pack.list | grep -o '"resident_bytes":[0-9]*' | cut -d: -f2)
if [ "$r3" -le $((32<<20)) ]; then echo "PASS pack-budget ($((r3>>20)) MiB <= 32)"; pass=$((pass+1)); else echo "FAIL pack-budget ($((r3>>20)) MiB)"; fail=$((fail+1)); fi
check pack-close    '"ok":true'                   "$CTL" pack.close 1

# --- workloads ------------------------------------------------------------
check spawn         '"running":true'              "$CTL" spawn sleeper -- sleep 30
check dup-spawn     'already running'             "$CTL" spawn sleeper -- sleep 30
if [ "$MODE" = systemd ]; then
  check cgroup-attach '"memory_current":'         "$CTL" procs
  # 300 MiB allocation against a 64 MiB memory.max must be OOM-killed inside its cgroup only.
  "$CTL" spawn hog --mem-max 64 --swap-max 0 -- python3 -c 'b=bytearray(300<<20); import time; time.sleep(5)' >/dev/null
  sleep 3
  check cgroup-oom    '"name":"hog","pid":[0-9]*,"running":false,"exit":-9' "$CTL" procs
  check daemon-alive  '"pong":true'                "$CTL" ping
  # Crash the broker: workloads must survive and be re-adopted by the restarted daemon.
  old=$("$CTL" stats | grep -o '"pid":[0-9]*' | cut -d: -f2)
  systemctl --user kill -s KILL --kill-whom=main rftmwd-test
  for _ in $(seq 50); do sleep 0.2; new=$("$CTL" stats 2>/dev/null | grep -o '"pid":[0-9]*' | cut -d: -f2); [ -n "$new" ] && [ "$new" != "$old" ] && break; done
  echo "daemon pid $old -> ${new:-none}"
  check readopt       '"name":"sleeper","pid":[0-9]*,"running":true,.*"adopted":true' "$CTL" procs
fi
# The workload inherits the caller's environment and working directory.
(cd "$T" && QOS_PROBE=inherited "$CTL" spawn envprobe -- sh -c 'echo "$QOS_PROBE $PWD" > probe.out') >/dev/null
sleep 1
check env-cwd-inherit "inherited $T"              cat "$T/probe.out"
check kill          '"ok":true'                   "$CTL" kill sleeper
sleep 1
check reaped        '"name":"sleeper","pid":[0-9]*,"running":false' "$CTL" procs

echo "---- $pass passed, $fail failed"
[ "$MODE" = systemd ] && journalctl --user -u rftmwd-test --no-pager -o cat 2>/dev/null | tail -20 || cat "$T/log" 2>/dev/null | tail -20
exit $((fail != 0))
