#!/usr/bin/env bash
# Run the Linux-side test suites (WSL or any systemd Linux). Tools fetched without
# root into ~/.local/qos-tools when the distro lacks them.
set -uo pipefail
L="$(cd "$(dirname "$0")/.." && pwd)"
TOOLS="$HOME/.local/qos-tools"
mkdir -p "$TOOLS"
fetch() { command -v "$1" >/dev/null || [ -x "$TOOLS/root/usr/bin/$1" ] || (cd "$TOOLS" && shift && apt-get download "$@" >/dev/null 2>&1 && for d in *.deb; do dpkg -x "$d" root; done); }
fetch nasm nasm
fetch shellcheck shellcheck
fetch jq jq libjq1 libonig5
export PATH="$TOOLS/root/usr/bin:$PATH" LD_LIBRARY_PATH="$TOOLS/root/usr/lib/x86_64-linux-gnu"
rc=0
[ $# -gt 0 ] || set -- asm build rftmwd session
for t in "$@"; do
    echo "######## $t"
    case $t in
        asm) bash "$L/tests/run_asm_abi_linux.sh" || rc=1 ;;
        build) bash "$L/tests/build_rftmwd_wsl.sh" || rc=1 ;;
        rftmwd) bash "$L/tests/test_rftmwd.sh" 2>&1 | grep "PASS\|FAIL\|----"; [ "${PIPESTATUS[0]}" = 0 ] || rc=1 ;;
        session) bash "$L/tests/test_session_tools.sh" || rc=1 ;;
        lint) shellcheck -S warning "$L"/*.sh "$L"/iso/*.sh "$L"/bin/* "$L"/tests/*.sh || rc=1 ;;
    esac
done
echo "######## overall: $([ $rc = 0 ] && echo PASS || echo FAIL)"
exit $rc
