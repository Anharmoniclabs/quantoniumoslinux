#!/usr/bin/env bash
# Boot a QuantoniumOS ISO headless in QEMU and check that the live session comes up
# with the RFTMW middleware. journald is forwarded to the serial console, so user-unit
# logs (rftmwd, qos-desktop) are visible without logging in.
#
#   boot_iso_qemu.sh /out/quantoniumos-*.iso [timeout_s]
set -uo pipefail
ISO="${1:?ISO}"; TIMEOUT="${2:-900}"
CPUS="${QOS_BOOT_CPUS:-4}"; MEMORY_MB="${QOS_BOOT_MEMORY_MB:-6144}"
W="$(mktemp -d)"; LOG="${QOS_BOOT_LOG:-/out/boot.log}"
trap 'rm -rf "$W"' EXIT
bsdtar -xf "$ISO" -C "$W" arch/boot/x86_64/vmlinuz-linux arch/boot/x86_64/initramfs-linux.img loader/entries
UUID="$(grep -ho 'archisosearchuuid=[^ ]*' "$W"/loader/entries/*.conf | head -1)"
ACCEL=tcg; [ -w /dev/kvm ] && ACCEL=kvm
echo "booting with $ACCEL, $UUID, log: $LOG"
timeout "$TIMEOUT" qemu-system-x86_64 -machine q35,accel=$ACCEL -cpu max -smp "$CPUS" -m "$MEMORY_MB" \
    -cdrom "$ISO" -kernel "$W/arch/boot/x86_64/vmlinuz-linux" -initrd "$W/arch/boot/x86_64/initramfs-linux.img" \
    -append "archisobasedir=arch $UUID console=ttyS0,115200 systemd.journald.forward_to_console=1 systemd.log_color=0" \
    -vga std -display none -serial "file:$LOG" -no-reboot &
QPID=$!
check() { grep -q "$1" "$LOG" 2>/dev/null; }
for _ in $(seq "$TIMEOUT"); do
    sleep 1
    kill -0 $QPID 2>/dev/null || break
    check "spawned shell pid" && check "listening on" && { sleep 60; break; }
done
kill $QPID 2>/dev/null; wait $QPID 2>/dev/null

rc=0
probe() { if check "$2"; then echo "PASS $1"; else echo "FAIL $1"; rc=1; fi; }
probe "kernel + archiso initramfs"     "running hook .archiso."
probe "reached multi-user"            "Reached target.*[Mm]ulti-[Uu]ser"
probe "zram swap"                     "zram0"
probe "greetd started"                "greetd"
probe "user session for qos"          "user@1000.service"
probe "rftmwd: scopes in rftmw.slice" "workloads run as systemd scopes in rftmw.slice"
probe "rftmwd: PSI pressure trigger"  "memory pressure: PSI trigger armed"
probe "rftmwd: socket listening"      "rftmwd: listening on /run/user/1000/rftmw/rftmwd.sock"
probe "session launched shell via qos-run" "spawned shell pid"
probe "Hyprland compositor up"        "Connected to a wayland compositor: QuantoniumOS:Hyprland"
probe "ollama served as workload"     "Listening on 127.0.0.1:11434"
probe "ollama: flash attention on"    "OLLAMA_FLASH_ATTENTION:true"
probe "ollama: q8_0 KV cache"         "OLLAMA_KV_CACHE_TYPE:q8_0"
if check "workload shell exited"; then echo "FAIL shell still running"; rc=1; else echo "PASS shell still running"; fi
echo "--- QuantoniumOS lines from the boot log"
grep -a "rftmwd\|qos-\|Hyprland\|greetd" "$LOG" | grep -v "^\s*$" | tail -25
exit $rc
