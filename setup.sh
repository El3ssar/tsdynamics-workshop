#!/usr/bin/env bash
# One-shot local setup for the TSDynamics workshop.
# Prefers uv (fast); falls back to the stdlib venv + pip. Requires Python >= 3.12.
set -euo pipefail
cd "$(dirname "$0")"

if command -v uv >/dev/null 2>&1; then
  echo ">> Using uv"
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt
else
  echo ">> uv not found; using python3 venv + pip"
  PY="$(command -v python3.12 || command -v python3)"
  "$PY" -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

echo
echo "Done. Launch the workshop with:"
echo "    ./.venv/bin/jupyter lab"
echo "and open notebooks/00_orientation.ipynb"
