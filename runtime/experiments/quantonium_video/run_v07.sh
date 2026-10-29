#!/usr/bin/env bash
# Blender image-to-rig prepass -> guided training -> registered frame rendering.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export QVIDEO_OUT="${QVIDEO_OUT:-$HOME/quantonium-video-runs/v07_codespace}"
export QVIDEO_PYTHON="${QVIDEO_PYTHON:-python}"
export QVIDEO_BLENDER="${QVIDEO_BLENDER:-blender}"
export QVIDEO_MOTION="${QVIDEO_MOTION:-$HOME/quantonium-video-runs/v04_codespace/latest.pt}"
if [[ ! -s "$QVIDEO_MOTION" && -s "$HOME/quantonium-video-runs/v04/latest.pt" ]]; then
  export QVIDEO_MOTION="$HOME/quantonium-video-runs/v04/latest.pt"
fi
export QVIDEO_STEPS="${QVIDEO_STEPS:-5000}"
export QVIDEO_THREADS="${QVIDEO_THREADS:-4}"
export QVIDEO_SAVE_EVERY="${QVIDEO_SAVE_EVERY:-250}"
export QVIDEO_SIZE="${QVIDEO_SIZE:-128}"
export QVIDEO_WIDTH="${QVIDEO_WIDTH:-16}"
export QVIDEO_BATCH="${QVIDEO_BATCH:-2}"
export QVIDEO_FRAMES="${QVIDEO_FRAMES:-72}"
export QVIDEO_RENDER_WIDTH="${QVIDEO_RENDER_WIDTH:-960}"
export QVIDEO_RENDER_HEIGHT="${QVIDEO_RENDER_HEIGHT:-540}"
OUT="$QVIDEO_OUT"
mkdir -p "$OUT"
PID="$OUT/run.pid"
LOG="$OUT/train.log"

running() {
  [[ -s "$PID" ]] || return 1
  local p proc_pid cmd
  read -r p proc_pid < "$PID"
  proc_pid="${proc_pid:-$p}"
  [[ "$p" =~ ^[0-9]+$ && "$proc_pid" =~ ^[0-9]+$ && -r "/proc/$proc_pid/cmdline" ]] || return 1
  kill -0 "$p" 2>/dev/null || return 1
  cmd="$(tr '\0' ' ' < "/proc/$proc_pid/cmdline")"
  [[ "$cmd" == *"run_v07.sh worker"* || "$cmd" == *"run_v07.sh run"* || "$cmd" == *"run_v07.sh render"* ]]
}

preflight() {
  for cmd in "$QVIDEO_PYTHON" "$QVIDEO_BLENDER" flock setsid; do
    command -v "$cmd" >/dev/null || { echo "Missing command: $cmd" >&2; return 2; }
  done
  "$QVIDEO_PYTHON" -c 'import torch, numpy, PIL' || {
    echo 'Activate your video Python environment or set QVIDEO_PYTHON.' >&2; return 2;
  }
  [[ -s "$QVIDEO_MOTION" ]] || { echo "Missing V0.4 checkpoint: $QVIDEO_MOTION" >&2; return 2; }
}

case "${1:-status}" in
  rig)
    [[ -n "${QVIDEO_IMAGE:-}" ]] || { echo 'Set QVIDEO_IMAGE to a full-body image.' >&2; exit 2; }
    exec "$QVIDEO_PYTHON" -u "$HERE/prepare_v07.py" --image "$QVIDEO_IMAGE" \
      --out "$OUT/image_rig" --blender "$QVIDEO_BLENDER" --size "$QVIDEO_SIZE" \
      --frames "$QVIDEO_FRAMES" --render-width "$QVIDEO_RENDER_WIDTH" \
      --render-height "$QVIDEO_RENDER_HEIGHT"
    ;;
  start)
    preflight
    exec 9>"$OUT/launch.lock"
    if ! flock -n 9; then echo 'Already running for this output directory'; exit 0; fi
    if [[ -f "$LOG" ]] && (( $(stat -c%s "$LOG") > 2097152 )); then
      tail -n 400 "$LOG" > "$LOG.tmp"
      mv "$LOG.tmp" "$LOG"
    fi
    rm -f "$OUT/exit_code"
    # Child inherits FD 9, holding the lock across Blender and training stages.
    nohup bash "$HERE/run_v07.sh" worker </dev/null >>"$LOG" 2>&1 &
    echo "Started Blender prepass + V0.7 training to $QVIDEO_STEPS TOTAL steps (PID $!)."
    echo "Log: $LOG"
    ;;
  run|render|worker)
    preflight
    mode="${1}"
    if [[ "$mode" != worker ]]; then
      exec 9>"$OUT/launch.lock"
      flock -n 9 || { echo 'Already running for this output directory' >&2; exit 2; }
    fi
    # Signal PID and /proc PID can differ in containers with an inherited procfs.
    read -r proc_pid proc_rest < /proc/self/stat
    echo "$$ $proc_pid" > "$PID"
    child=0
    cleanup() { local rc=$?; echo "$rc" > "$OUT/exit_code"; rm -f "$PID"; }
    stop_child() {
      trap - TERM INT
      if (( child > 0 )); then
        kill -TERM -- "-$child" 2>/dev/null || true
        wait "$child" || true
      fi
      exit 143
    }
    trap cleanup EXIT
    trap stop_child TERM INT
    common=(--size "$QVIDEO_SIZE" --frames "$QVIDEO_FRAMES" --fps 24
            --render-width "$QVIDEO_RENDER_WIDTH" --render-height "$QVIDEO_RENDER_HEIGHT"
            --motion-checkpoint "$QVIDEO_MOTION")
    setsid "$QVIDEO_PYTHON" -u "$HERE/prepare_v07.py" "${common[@]}" \
      --blender "$QVIDEO_BLENDER" --out "$OUT/blender_shot" &
    child=$!
    wait "$child"
    child=0
    extra=()
    [[ "$mode" == render ]] && extra+=(--preview-only)
    setsid "$QVIDEO_PYTHON" -u "$HERE/train_v07.py" "${common[@]}" \
      --shot "$OUT/blender_shot" --out "$OUT" --steps "$QVIDEO_STEPS" \
      --threads "$QVIDEO_THREADS" --save-every "$QVIDEO_SAVE_EVERY" \
      --width "$QVIDEO_WIDTH" --batch "$QVIDEO_BATCH" "${extra[@]}" &
    child=$!
    wait "$child"
    child=0
    ;;
  status)
    if running; then read -r p proc_pid < "$PID"; echo "RUNNING (PID $p)"; else echo 'NOT RUNNING'; fi
    if [[ -f "$OUT/exit_code" ]]; then echo "Last exit code: $(cat "$OUT/exit_code")"; fi
    [[ ! -f "$LOG" ]] || tail -n 12 "$LOG"
    ;;
  logs)
    touch "$LOG"
    tail -n 30 -f "$LOG"
    ;;
  stop)
    if running; then
      read -r p proc_pid < "$PID"
      kill -TERM "$p"
      echo 'Stop requested; training saves after its current optimizer step.'
    else echo 'Not running'; fi
    ;;
  *) echo 'Usage: bash run_v07.sh start|run|render|rig|status|logs|stop' >&2; exit 2;;
esac
