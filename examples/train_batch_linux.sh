#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root. Optional: DATA_DIR=/path/to/images RUN=/path/to/new/run.
DATA_DIR="${DATA_DIR:-/path/to/images}"
RUN="${RUN:-runs/adasplat_$(date +%Y%m%d_%H%M%S)}"
LOG="${RUN}.launcher.log"
mkdir -p "$(dirname "$RUN")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
nohup python -u train_batch.py \
  --config configs/default.json \
  --input-dir "$DATA_DIR" --recursive \
  --output "$RUN" --gpus 0,1 --jobs-per-gpu 2 \
  --num 200000 --initial-ratio 0.15 --steps 50000 \
  --save-counts 40000,50000,70000,100000,150000,200000 \
  --no-early-stop \
  > "$LOG" 2>&1 < /dev/null &
echo "PID=$!"
echo "OUTPUT=$RUN"
echo "LOG=$LOG"
