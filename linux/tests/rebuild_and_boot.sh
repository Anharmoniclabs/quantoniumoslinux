#!/usr/bin/env bash
# Rebuild the ISO in the WSL Arch chroot, then boot-test it. Run as root.
set -uo pipefail
T="$(cd "$(dirname "$0")" && pwd)"
bash "$T/wsl_arch_iso.sh" > /var/tmp/qos-iso-build.log 2>&1 || { echo "ISO build failed"; tail -30 /var/tmp/qos-iso-build.log; exit 1; }
grep -a "INFO: Done!$" /var/tmp/qos-iso-build.log | tail -1; ls -la /var/tmp/qos-arch/out/*.iso
bash "$T/wsl_arch_run.sh" "rm -f /out/boot.log; bash /src/linux/tests/boot_iso_qemu.sh /out/quantoniumos-\$(date +%Y.%m.%d)-x86_64.iso 1500"
