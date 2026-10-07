#!/usr/bin/env bash
# One-command verification. Usage: ./verify.sh [--browser]
set -euo pipefail
cd "$(dirname "$0")"
# Default: the monorepo uv environment (`uv sync --all-packages` at the repo root).
# Fallback: FSL's python, which finds neuro_core through tractlab/__init__.py.
DEFAULT_PY="../.venv/bin/python"
[[ -x "$DEFAULT_PY" ]] || DEFAULT_PY="$HOME/fsl/bin/python"
PY="${TRACTLAB_PYTHON:-$DEFAULT_PY}"
[[ -x "$PY" ]] || { echo "python not found at $PY (set TRACTLAB_PYTHON)"; exit 2; }
echo "== python: $PY"

[[ "$(git config core.hooksPath || true)" == "hooks" ]] || echo "WARN: PHI hook not installed — run scripts/install_hooks.sh"

echo "== python"
"$PY" -m pytest -q -p no:warnings -ra | tee /tmp/tractlab-pytest.log
skipped=$(grep -Eo '[0-9]+ skipped' /tmp/tractlab-pytest.log | grep -Eo '[0-9]+' || echo 0)
echo "== python skipped: $skipped (private case + MRtrix absent => large; see CLAUDE.md)"

echo "== js"
npm test --silent

if [[ "${1:-}" == "--browser" ]]; then
  echo "== browser"
  npm run test:browser
  # Demo-case gates: start the demo server first, e.g.
  # Visual baselines live in ~/.cache/tractlab/visual-baselines (never in git); first run: npm run test:visual -- --update
  #   TRACTLAB_MANIFEST="$PWD/cases/demo-leipzig-sub-010005/manifest.json" ./serve.sh 18995 --background
  npm run test:pick -- --url "${TRACTLAB_DEMO_URL:-http://127.0.0.1:18995/index.html}"
  npm run test:visual -- --url "${TRACTLAB_DEMO_URL:-http://127.0.0.1:18995/index.html}"
fi
echo "== OK"
