#!/usr/bin/env bash
# Run a command inside the WSL Arch chroot made by wsl_arch_iso.sh (as root), with the
# checkout at /src (read-only) and ISO output at /out.
#   wsl_arch_run.sh 'bash /src/linux/tests/inspect_iso.sh /out/quantoniumos-*.iso'
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="${QOS_ARCH_BASE:-/var/tmp/qos-arch}"; R="$BASE/root.x86_64"
mountpoint -q "$R" || mount --bind "$R" "$R"
mountpoint -q "$R/src" || mount --bind -o ro "$SRC" "$R/src"
mountpoint -q "$R/out" || mount --bind "$BASE/out" "$R/out"
# shellcheck disable=SC2154  # rc is assigned inside the trap string
trap 'rc=$?; umount "$R/src" "$R/out" 2>/dev/null; exit $rc' EXIT
"$R/usr/bin/arch-chroot" "$R" /bin/bash -c "$1"
