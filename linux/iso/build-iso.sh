#!/usr/bin/env bash
# Build a bootable QuantoniumOS live ISO (UEFI + BIOS) with archiso.
#
#   sudo linux/iso/build-iso.sh [OUT_DIR]
#
# Runs on Arch Linux (or an Arch container/chroot) as root. The live system boots
# straight into the QuantoniumOS session as user "qos" (password "qos"), with the
# full stack prebuilt: rftmwd, rftmw_native kernels/CLIs, the Tauri shell.
#
# Environment:
#   QOS_WORK=/var/tmp/qos-iso    scratch space (needs ~15 GB)
#   QOS_SKIP_DESKTOP=1           headless ISO without the Tauri shell (CI smoke builds)
#   QOS_BUILD_USER=<user>        unprivileged user for compiling (default: $SUDO_USER or "nobody")
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
OUT="$(mkdir -p "${1:-$SRC/linux/iso/out}" && cd "${1:-$SRC/linux/iso/out}" && pwd)"
WORK="${QOS_WORK:-/var/tmp/qos-iso}"
BUILD_USER="${QOS_BUILD_USER:-${SUDO_USER:-nobody}}"
LIVE_USER=qos
step() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }

[ "$(id -u)" -eq 0 ] || { echo "run as root (mkarchiso needs it)" >&2; exit 1; }
command -v mkarchiso >/dev/null || pacman -S --needed --noconfirm archiso
mapfile -t PKGS < <(sed 's/#.*//; /^\s*$/d' "$SRC/linux/packages.txt")

step "Build dependencies on the host"
pacman -S --needed --noconfirm "${PKGS[@]}" rsync

step "Building QuantoniumOS (x86-64-v3 baseline) as $BUILD_USER"
STAGE="$WORK/stage"
rm -rf "$STAGE"; install -d "$STAGE"
rsync -a --exclude '.git' --exclude 'node_modules' --exclude '.venv' \
    --exclude 'src-tauri/target' --exclude 'rftmw_native/build' --exclude 'rftmw_native/build_asm' \
    --exclude '__pycache__' --exclude 'data/browser-test' --exclude '*.bak' --exclude 'iso/out' \
    "$SRC/runtime" "$SRC/linux" "$STAGE/"
chown -R "$BUILD_USER" "$STAGE"
# HOME must be writable for cargo/npm caches when building as an arbitrary user.
install -d -o "$BUILD_USER" "$WORK/home"
runuser -u "$BUILD_USER" -- env HOME="$WORK/home" QOS_MARCH=x86-64-v3 QOS_SKIP_DESKTOP="${QOS_SKIP_DESKTOP:-0}" \
    "$STAGE/linux/build.sh" "$STAGE"

step "Assembling archiso profile"
PROFILE="$WORK/profile"
rm -rf "$PROFILE"
cp -r /usr/share/archiso/configs/releng "$PROFILE"
AIR="$PROFILE/airootfs"
{ cat "$PROFILE/packages.x86_64"; printf '%s\n' "${PKGS[@]}" rsync; } | sort -u > "$PROFILE/packages.x86_64.new"
mv "$PROFILE/packages.x86_64.new" "$PROFILE/packages.x86_64"

sed -i \
    -e 's/^iso_name=.*/iso_name="quantoniumos"/' \
    -e 's/^iso_label=.*/iso_label="QOS_$(date --date="@${SOURCE_DATE_EPOCH:-$(date +%s)}" +%Y%m)"/' \
    -e 's/^iso_publisher=.*/iso_publisher="QuantoniumOS"/' \
    -e 's/^iso_application=.*/iso_application="QuantoniumOS Live"/' \
    "$PROFILE/profiledef.sh"

# releng autologins root on tty1; greetd owns tty1 here.
rm -rf "$AIR/etc/systemd/system/getty@tty1.service.d"
# releng ships systemd-networkd + iwd; QuantoniumOS uses NetworkManager.
rm -f "$AIR/etc/systemd/system/multi-user.target.wants/systemd-networkd.service" \
      "$AIR/etc/systemd/system/multi-user.target.wants/iwd.service" \
      "$AIR/etc/systemd/system/sockets.target.wants/systemd-networkd.socket" \
      "$AIR/etc/systemd/system/systemd-networkd-wait-online.service.d" 2>/dev/null || true

"$STAGE/linux/install-files.sh" "$AIR" "$STAGE" "$LIVE_USER"
ln -sf /usr/lib/systemd/system/NetworkManager.service "$AIR/etc/systemd/system/multi-user.target.wants/NetworkManager.service"
echo 'w /sys/module/zswap/parameters/enabled - - - - N' > "$AIR/etc/tmpfiles.d/91-quantoniumos-zswap.conf"

# Live user "qos" (uid 1000), password "qos", passwordless sudo on the live medium.
HASH="$(openssl passwd -6 qos)"
# releng ships these with root's entry; seed them if a future releng does not.
[ -f "$AIR/etc/passwd" ] || echo "root:x:0:0:root:/root:/usr/bin/zsh" > "$AIR/etc/passwd"
[ -f "$AIR/etc/shadow" ] || echo "root::14871::::::" > "$AIR/etc/shadow"
[ -f "$AIR/etc/group" ] || echo "root:x:0:root" > "$AIR/etc/group"
[ -f "$AIR/etc/gshadow" ] || echo "root:::root" > "$AIR/etc/gshadow"
echo "$LIVE_USER:x:1000:1000:QuantoniumOS:/home/$LIVE_USER:/bin/bash" >> "$AIR/etc/passwd"
echo "$LIVE_USER:$HASH:19000:0:99999:7:::" >> "$AIR/etc/shadow"
echo "$LIVE_USER:x:1000:" >> "$AIR/etc/group"
echo "$LIVE_USER:!*::" >> "$AIR/etc/gshadow"
install -d -m 0750 "$AIR/etc/sudoers.d"
echo "$LIVE_USER ALL=(ALL:ALL) NOPASSWD: ALL" > "$AIR/etc/sudoers.d/10-quantoniumos-live"
install -d "$AIR/home/$LIVE_USER/.config"
cp -r "$AIR/etc/skel/.config/." "$AIR/home/$LIVE_USER/.config/"
# The live user owns the tree (the shell writes caches/state beside it).
cat >> "$PROFILE/profiledef.sh" <<'EOF'
file_permissions+=(
  ["/etc/shadow"]="0:0:400"
  ["/etc/gshadow"]="0:0:400"
  ["/etc/sudoers.d"]="0:0:750"
  ["/etc/sudoers.d/10-quantoniumos-live"]="0:0:440"
  ["/home/qos"]="1000:1000:750"
  ["/opt/quantoniumos"]="1000:1000:755"
  ["/usr/lib/quantoniumos/rftmwd"]="0:0:755"
  ["/usr/bin/rftmwctl"]="0:0:755"
  ["/usr/bin/qos-run"]="0:0:755"
  ["/usr/bin/qos-session"]="0:0:755"
  ["/usr/bin/qos-desktop"]="0:0:755"
  ["/usr/bin/qos-status"]="0:0:755"
)
EOF
# mkarchiso copies airootfs without preserving modes: every executable must be listed,
# or the shell, native CLIs and venv tools lose +x on the live system.
(cd "$AIR" && find opt/quantoniumos -type f -perm -u+x) | {
    echo 'file_permissions+=('
    while read -r f; do printf '  ["/%s"]="1000:1000:755"\n' "$f"; done
    echo ')'
} >> "$PROFILE/profiledef.sh"
# file_permissions is not recursive; fix ownership of the home and tree contents at boot.
cat > "$AIR/etc/tmpfiles.d/92-quantoniumos-live.conf" <<EOF
Z /home/$LIVE_USER - $LIVE_USER $LIVE_USER -
Z /opt/quantoniumos - $LIVE_USER $LIVE_USER -
EOF

# Boot menu branding.
find "$PROFILE" \( -path '*/syslinux/*.cfg' -o -path '*/efiboot/loader/entries/*.conf' -o -path '*/grub/*.cfg' \) \
    -exec sed -i 's/Arch Linux install medium/QuantoniumOS live/g; s/Arch Linux/QuantoniumOS/g' {} +

step "mkarchiso"
rm -rf "$WORK/work"
mkarchiso -v -w "$WORK/work" -o "$OUT" "$PROFILE"
ls -la "$OUT"/quantoniumos-*.iso
