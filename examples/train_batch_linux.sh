#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${DATA_DIR:?Set DATA_DIR to your image directory}"
RUN="${RUN:-runs/batch_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname "$RUN")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
nohup python -u train_batch.py --input-dir "$DATA_DIR" --output "$RUN" --encode > "${RUN}.launcher.log" 2>&1 < /dev/null &
echo "PID=$!"
echo "LOG=${RUN}.launcher.log"
