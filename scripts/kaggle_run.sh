#!/bin/bash
# Kaggle run: execute the broad-validation benchmark.
# Results are written to /kaggle/working/results.json (downloadable from Kaggle).
set -euo pipefail

python experiments/exp_broad_validation.py \
    --n 300 \
    --seeds 1 2 3
echo "[kaggle_run] done"
