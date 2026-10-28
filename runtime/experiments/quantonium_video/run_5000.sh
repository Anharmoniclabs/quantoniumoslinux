#!/usr/bin/env bash
# Launch and monitor a bounded 5,000-total-step V0.2 training run in WSL.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="${QVIDEO_TRAINER:-$HERE/train_v02.py}"
PYTHON="${QVIDEO_PYTHON:-$HOME/.quantonium-video-env/bin/python}"
OUT="${QVIDEO_OUT:-$HOME/quantonium-video-runs/v02}"
TARGET="${QVIDEO_STEPS:-5000}"
THREADS="${QVIDEO_THREADS:-4}"
SAVE_EVERY="${QVIDEO_SAVE_EVERY:-200}"
PID_FILE="$OUT/run_5000.pid"
LOG_FILE="$OUT/train_5000.log"
mkdir -p "$OUT"

running() {
    [[ -s "$PID_FILE" ]] || return 1
    local pid cmdline
    pid="$(cat "$PID_FILE")"
    [[ "$pid" =~ ^[0-9]+$ ]] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    [[ -r "/proc/$pid/cmdline" ]] || return 1
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    [[ "$cmdline" == *"run_5000.sh worker"* ]]
}

case "${1:-status}" in
    start)
        [[ -x "$PYTHON" ]] || { echo "Missing Python environment: $PYTHON" >&2; exit 1; }
        [[ -f "$TRAINER" ]] || { echo "Missing trainer: $TRAINER" >&2; exit 1; }
        if running; then echo "Already running (PID $(cat "$PID_FILE"))"; exit 0; fi
        rm -f "$PID_FILE"
        if [[ -f "$LOG_FILE" ]] && (( $(stat -c%s "$LOG_FILE") > 1048576 )); then
            tail -n 400 "$LOG_FILE" > "$LOG_FILE.trim" && mv "$LOG_FILE.trim" "$LOG_FILE"
        fi
        nohup bash "$HERE/run_5000.sh" worker </dev/null >>"$LOG_FILE" 2>&1 &
        echo "$!" > "$PID_FILE"
        echo "Started video training ($TRAINER) to $TARGET TOTAL steps (PID $(cat "$PID_FILE"))."
        echo "Checkpoint: $OUT/latest.pt"
        echo "Log: $LOG_FILE"
        echo "Use '$0 status' or '$0 logs' to inspect progress."
        ;;
    worker)
        stop_requested=0
        child_pid=0
        on_stop() {
            stop_requested=1
            if (( child_pid > 0 )); then kill -TERM "$child_pid" 2>/dev/null || true; fi
        }
        trap on_stop TERM INT
        retries=0
        while (( !stop_requested )); do
            echo "=== $(date -Is) training to $TARGET total steps ==="
            "$PYTHON" -u "$TRAINER" --out "$OUT" --steps "$TARGET" \
                --save-every "$SAVE_EVERY" --threads "$THREADS" &
            child_pid=$!
            if wait "$child_pid"; then rc=0; else rc=$?; fi
            child_pid=0
            if (( stop_requested )); then echo "Stopped by request"; break; fi
            if (( rc == 0 )); then echo "Training process completed"; break; fi
            retries=$(( retries + 1 ))
            echo "Training exited $rc; restart $retries of 3" >&2
            if (( retries >= 3 )); then echo "Too many failures; inspect log" >&2; break; fi
            sleep 5
        done
        if [[ "$(cat "$PID_FILE" 2>/dev/null || true)" == "$$" ]]; then rm -f "$PID_FILE"; fi
        ;;
    status)
        if running; then echo "RUNNING (PID $(cat "$PID_FILE"))"; else echo "NOT RUNNING"; fi
        [[ -f "$OUT/latest.pt" ]] && echo "Checkpoint: $OUT/latest.pt"
        [[ -f "$LOG_FILE" ]] && tail -n 12 "$LOG_FILE"
        ;;
    stop)
        if running; then
            kill -TERM "$(cat "$PID_FILE")"
            echo "Stop requested; trainer will save a checkpoint before exiting."
        else echo "Not running"; fi
        ;;
    logs)
        [[ -f "$LOG_FILE" ]] || { echo "No log yet: $LOG_FILE"; exit 1; }
        tail -n 30 -f "$LOG_FILE"
        ;;
    *) echo "Usage: bash $0 {start|status|stop|logs}" >&2; exit 2 ;;
esac
