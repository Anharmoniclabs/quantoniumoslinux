#!/usr/bin/env bash
# Inspect a built QuantoniumOS ISO: list key files inside its squashfs and show the
# session/service wiring. Run as root inside the Arch chroot (needs squashfs-tools).
#
#   inspect_iso.sh /out/quantoniumos-*.iso
set -euo pipefail
ISO="${1:?ISO path}"
M="$(mktemp -d)"; X="$(mktemp -d)"
trap 'umount "$M" 2>/dev/null; rm -rf "$M" "$X"' EXIT
mount -o loop,ro "$ISO" "$M" 2>/dev/null || { command -v bsdtar >/dev/null && bsdtar -xf "$ISO" -C "$M" arch/x86_64/airootfs.sfs; }
SFS="$M/arch/x86_64/airootfs.sfs"
unsquashfs -q -d "$X/r" "$SFS" \
    usr/lib/quantoniumos usr/bin/rftmwctl usr/bin/qos-run usr/bin/qos-session usr/bin/qos-desktop usr/bin/qos-status \
    opt/quantoniumos/runtime/desktop-shell/src-tauri/target/release/quantoniumos-desktop \
    'opt/quantoniumos/runtime/src/rftmw_native/build/*.so' 'opt/quantoniumos/runtime/src/rftmw_native/build/rftmw_*' \
    etc/systemd/user etc/systemd/system/display-manager.service etc/systemd/system/user@.service.d \
    usr/lib/systemd/user/rftmwd.service usr/lib/systemd/user/rftmw.slice etc/greetd/config.toml \
    etc/sysctl.d/90-quantoniumos.conf etc/systemd/zram-generator.conf home/qos/.config/hypr etc/passwd \
    usr/bin/Hyprland usr/bin/ollama usr/bin/tuigreet \
    usr/share/quantoniumos/rftmw-basis etc/quantoniumos etc/tmpfiles.d etc/udev/rules.d >/dev/null
cd "$X/r"
find . -type f -o -type l | sort | sed 's/^\.//'
echo "--- greetd initial_session"; grep -A2 initial_session etc/greetd/config.toml
echo "--- live user";             grep '^qos:' etc/passwd
echo "--- display manager";       readlink etc/systemd/system/display-manager.service
echo "--- rftmwd enabled";        readlink etc/systemd/user/default.target.wants/rftmwd.service
echo "--- rftmwd binary";         file -b usr/lib/quantoniumos/rftmwd | cut -c1-80
echo "--- prebuilt RFT bases";    echo usr/share/quantoniumos/rftmw-basis/*.bin
echo "--- ollama tuning";         grep -v '^#' etc/quantoniumos/ollama.env | grep .
echo "--- MGLRU / readahead";     grep -h '^w' etc/tmpfiles.d/93-quantoniumos-mglru.conf; grep -o 'read_ahead_kb}="[0-9]*"' etc/udev/rules.d/60-quantoniumos-readahead.rules
