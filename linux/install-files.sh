#!/usr/bin/env bash
# Install a built QuantoniumOS tree into a root filesystem.
#
#   install-files.sh ROOT TREE USER
#
#   ROOT  target root: "/" for a real install, archiso's airootfs for the ISO
#   TREE  built QuantoniumOS tree (see build.sh); copied to ROOT/opt/quantoniumos
#   USER  account that autologins into the QuantoniumOS session
#
# Writes only under ROOT. Enabling services is done with symlinks so the same
# code path works for a live system and for an image that has never booted.
set -euo pipefail

ROOT="${1:?ROOT}"; TREE="$(cd "${2:?TREE}" && pwd)"; USER_NAME="${3:?USER}"
LINUX="$TREE/linux"
OPT="$ROOT/opt/quantoniumos"

install -d "$OPT" "$ROOT/usr/bin" "$ROOT/usr/lib/quantoniumos"

# 1. The QuantoniumOS tree (sources + built artifacts) at /opt/quantoniumos.
if [ "$TREE" != "$(cd "$OPT" && pwd)" ]; then
    rsync -a --delete \
        --exclude '.git' --exclude 'node_modules' --exclude 'src-tauri/target/debug' \
        --exclude 'src-tauri/target/release/build' --exclude 'src-tauri/target/release/deps' \
        --exclude 'src-tauri/target/release/incremental' --exclude '__pycache__' \
        --exclude 'data/browser-test' --exclude '*.bak' \
        "$TREE/runtime" "$TREE/linux" "$TREE/build" "$OPT/"
fi

# 2. Middleware + session binaries.
install -m 0755 "$TREE/build/rftmwd/rftmwd" "$ROOT/usr/lib/quantoniumos/rftmwd"
install -m 0755 "$TREE/build/rftmwd/rftmwctl" "$ROOT/usr/bin/rftmwctl"
for f in "$LINUX"/bin/*; do install -m 0755 "$f" "$ROOT/usr/bin/$(basename "$f")"; done
for f in "$TREE"/runtime/src/rftmw_native/build/rftmw_*; do
    [ -f "$f" ] && [ -x "$f" ] && ln -sf "/opt/quantoniumos/runtime/src/rftmw_native/build/$(basename "$f")" "$ROOT/usr/bin/$(basename "$f")"
done

# 3. System configuration (systemd units, sysctl, zram, THP, greetd, session file).
#    Existing files are kept as *.pre-quantoniumos once, so the install is reversible.
(cd "$LINUX/system" && find . -type f) | while read -r rel; do
    dst="$ROOT/${rel#./}"
    install -d "$(dirname "$dst")"
    if [ -e "$dst" ] && [ ! -e "$dst.pre-quantoniumos" ] && ! cmp -s "$LINUX/system/$rel" "$dst"; then
        cp -a "$dst" "$dst.pre-quantoniumos"
    fi
    install -m 0644 "$LINUX/system/$rel" "$dst"
done
sed -i "s/@USER@/$USER_NAME/" "$ROOT/etc/greetd/config.toml"

# 4. Services: greetd as display manager, rftmwd for every user session.
WANTS="$ROOT/etc/systemd/system"
install -d "$WANTS/multi-user.target.wants" "$ROOT/etc/systemd/user/default.target.wants"
ln -sf /usr/lib/systemd/system/greetd.service "$WANTS/display-manager.service"
ln -sf /usr/lib/systemd/user/rftmwd.service "$ROOT/etc/systemd/user/default.target.wants/rftmwd.service"
for svc in NetworkManager bluetooth; do
    [ -e "$ROOT/usr/lib/systemd/system/$svc.service" ] && ln -sf "/usr/lib/systemd/system/$svc.service" "$WANTS/multi-user.target.wants/$svc.service"
done

# 5. Desktop configuration for new users (skel) and for USER if the home exists.
SKEL="$ROOT/etc/skel/.config"
install -d "$SKEL"
cp -r "$LINUX/config/hypr" "$LINUX/config/waybar" "$SKEL/"
echo "QuantoniumOS files installed into $ROOT (session user: $USER_NAME)"
