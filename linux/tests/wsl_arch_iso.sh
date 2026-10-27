#!/usr/bin/env bash
# Build the QuantoniumOS ISO on a non-Arch Linux host (e.g. WSL Ubuntu) inside an
# Arch bootstrap chroot. Run as root. Everything lives under $BASE; delete it to undo.
#
#   sudo linux/tests/wsl_arch_iso.sh            -> $BASE/out/quantoniumos-*.iso
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
BASE="${QOS_ARCH_BASE:-/var/tmp/qos-arch}"
ROOT="$BASE/root.x86_64"
MIRROR="${QOS_ARCH_MIRROR:-https://geo.mirror.pkgbuild.com}"
log() { printf '\n[%(%H:%M:%S)T] %s\n' -1 "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root" >&2; exit 1; }
mkdir -p "$BASE"

if [ ! -x "$ROOT/usr/bin/pacman" ]; then
    log "Fetching Arch bootstrap"
    if ! command -v zstd >/dev/null; then
        # Unpack a zstd binary next to the chroot rather than installing into the host.
        (mkdir -p "$BASE/tools" && cd "$BASE/tools" && apt-get download zstd >/dev/null && dpkg -x zstd_*.deb .)
        export PATH="$BASE/tools/usr/bin:$PATH"
    fi
    curl -fL --retry 3 -o "$BASE/bootstrap.tar.zst" "$MIRROR/iso/latest/archlinux-bootstrap-x86_64.tar.zst"
    tar -x --zstd -f "$BASE/bootstrap.tar.zst" -C "$BASE" --numeric-owner
    echo "Server = $MIRROR/\$repo/os/\$arch" > "$ROOT/etc/pacman.d/mirrorlist"
    sed -i 's/^#\?ParallelDownloads.*/ParallelDownloads = 8/' "$ROOT/etc/pacman.conf"
    # The bootstrap has no /etc/resolv.conf; arch-chroot bind-mounts the host's.
fi

# Source checkout inside the chroot (read-only bind); build output goes to /out.
mkdir -p "$ROOT/src" "$ROOT/out" "$BASE/out"
# arch-chroot requires the root to be a mountpoint. Bind it first: a later self-bind
# would hide the /src and /out mounts below it.
mountpoint -q "$ROOT" || mount --bind "$ROOT" "$ROOT"
mountpoint -q "$ROOT/src" || mount --bind -o ro "$SRC" "$ROOT/src"
mountpoint -q "$ROOT/out" || mount --bind "$BASE/out" "$ROOT/out"
cleanup() { umount "$ROOT/src" "$ROOT/out" 2>/dev/null || true; }
# shellcheck disable=SC2154  # rc is assigned inside the trap string
trap 'rc=$?; cleanup; exit $rc' EXIT
[ -f "$ROOT/src/linux/iso/build-iso.sh" ] || { echo "source not visible in chroot" >&2; exit 1; }

log "Preparing chroot"
"$ROOT/usr/bin/arch-chroot" "$ROOT" /bin/bash -euo pipefail -c '
    if [ ! -d /etc/pacman.d/gnupg ]; then pacman-key --init; pacman-key --populate archlinux; fi
    pacman -Syu --needed --noconfirm archiso arch-install-scripts base-devel rsync openssl
    id builder >/dev/null 2>&1 || useradd -m builder
'

log "Building ISO"
"$ROOT/usr/bin/arch-chroot" "$ROOT" /bin/bash -euo pipefail -c \
    "QOS_BUILD_USER=builder QOS_WORK=/var/tmp/qos-iso QOS_SKIP_DESKTOP=${QOS_SKIP_DESKTOP:-0} bash /src/linux/iso/build-iso.sh /out"
log "Done"
ls -la "$BASE/out"
