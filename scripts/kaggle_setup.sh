#!/bin/bash
# Kaggle setup: upgrade pip and install dependencies.
# The cog package is pure-stdlib, so requirements.txt is intentionally minimal.
set -euo pipefail

python -m pip install --upgrade pip
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi
echo "[kaggle_setup] done"
