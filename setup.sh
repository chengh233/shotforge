#!/usr/bin/env bash
# Install shotforge dependencies on the GPU box (Colab / L4).
# Intentionally does NOT touch torch — Colab ships it preinstalled.
set -e

pip install -r requirements.txt

echo
echo "[setup] done. Render the example with:"
echo "    python -m shotforge.generate --project projects/example"
