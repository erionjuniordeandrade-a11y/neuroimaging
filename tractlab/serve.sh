#!/usr/bin/env bash
# Launch TractLab loopback viewer. Usage: ./serve.sh [port] [--background|--stop]
set -euo pipefail
PORT=18993
if [[ $# -gt 0 && "$1" != --* ]]; then
  PORT="$1"
  shift
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--background" && "$1" != "--stop" ) ]]; then
  echo "Usage: ./serve.sh [port] [--background|--stop]" >&2
  exit 2
fi
cd "$(dirname "$0")"
PY="${TRACTLAB_PYTHON:-$HOME/fsl/bin/python}"
[[ -x "$PY" ]] || {
  echo "TractLab cannot start: Python interpreter unavailable (set TRACTLAB_PYTHON)." >&2
  exit 2
}
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" run.py "$PORT" "$@"
