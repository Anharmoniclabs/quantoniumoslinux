#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../../.." && pwd)"

# Prefer the user's active environment, then the repository's Codespace venv.
if [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -x "$repo_root/.phistate-env/bin/python" ]]; then
    python_bin="$repo_root/.phistate-env/bin/python"
else
    python_bin="python3"
fi

if ! "$python_bin" -c 'import numpy; import matplotlib' >/dev/null 2>&1; then
    "$python_bin" -m pip install -r "$script_dir/requirements.txt"
fi

exec "$python_bin" "$script_dir/rft_fibonacci_prefix.py" "$@"
