#!/usr/bin/env bash
# QuantoniumOS installer (Omarchy-style): turns a fresh Arch Linux install into
# QuantoniumOS. Run as your normal user (with sudo rights) from the checkout:
#
#   ./linux/install.sh
#
# Steps: packages -> /opt/quantoniumos tree -> build (rftmwd, rftmw_native, shell)
# -> system files & services -> desktop config -> reboot into QuantoniumOS.
# Re-running is safe; it updates in place. Replaced system files are kept as
# *.pre-quantoniumos.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
OPT=/opt/quantoniumos
step() { printf '\n\033[1;35m==> %s\033[0m\n' "$*"; }

[ -f /etc/arch-release ] || { echo "QuantoniumOS installs on Arch Linux (see linux/README.md)." >&2; exit 1; }
[ "$(id -u)" -ne 0 ] || { echo "Run as your normal user; sudo is used where needed." >&2; exit 1; }
grep -qw avx2 /proc/cpuinfo && grep -qw fma /proc/cpuinfo || echo "warning: CPU lacks AVX2/FMA; RFTMW ASM kernels will be skipped at runtime." >&2

step "Installing packages"
mapfile -t PKGS < <(sed 's/#.*//; /^\s*$/d' "$HERE/packages.txt")
sudo pacman -Syu --needed --noconfirm "${PKGS[@]}" rsync

step "Staging QuantoniumOS tree at $OPT"
sudo install -d -o "$USER" -g "$(id -gn)" "$OPT"
rsync -a --delete --exclude '.git' --exclude 'node_modules' --exclude '.venv' \
    --exclude 'src-tauri/target' --exclude 'rftmw_native/build' --exclude 'rftmw_native/build_asm' \
    --exclude '__pycache__' --exclude 'data/browser-test' --exclude '*.bak' \
    "$SRC/runtime" "$SRC/linux" "$OPT/"

step "Building the QuantoniumOS stack"
QOS_MARCH="${QOS_MARCH:-native}" "$OPT/linux/build.sh" "$OPT"

step "Installing system files and services"
sudo "$OPT/linux/install-files.sh" / "$OPT" "$USER"
sudo systemctl daemon-reload
sudo sysctl --system >/dev/null
sudo systemd-tmpfiles --create /etc/tmpfiles.d/90-quantoniumos-thp.conf || true
# zram replaces zswap; compressing twice wastes CPU.
echo N | sudo tee /sys/module/zswap/parameters/enabled >/dev/null 2>&1 || true
echo 'w /sys/module/zswap/parameters/enabled - - - - N' | sudo tee /etc/tmpfiles.d/91-quantoniumos-zswap.conf >/dev/null
sudo systemctl enable greetd.service NetworkManager.service bluetooth.service
sudo systemctl --global enable rftmwd.service

step "Desktop configuration for $USER"
for d in hypr waybar; do
    if [ -d "$HOME/.config/$d" ] && [ ! -d "$HOME/.config/$d.pre-quantoniumos" ]; then
        cp -a "$HOME/.config/$d" "$HOME/.config/$d.pre-quantoniumos"
    fi
    install -d "$HOME/.config/$d"
    for f in "$OPT/linux/config/$d"/*; do
        # Never clobber personal overrides.
        [ "$(basename "$f")" = user.conf ] && [ -e "$HOME/.config/$d/user.conf" ] && continue
        install -m 0644 "$f" "$HOME/.config/$d/"
    done
done

step "Checking the middleware"
systemctl --user daemon-reload
systemctl --user restart rftmwd.service
sleep 1
rftmwctl ping && rftmwctl rft.selftest 64

step "QuantoniumOS installed"
echo "Reboot to start QuantoniumOS (greetd logs $USER straight into the session)."
echo "Import your Ollama model (default 'eldrae:smartweights') with: ollama create eldrae:smartweights -f <Modelfile>"
