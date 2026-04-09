#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-nsys_mnist}
EPOCHS=${2:-1}
BATCH_SIZE=${3:-128}
MAX_STEPS=${4:-200}
DATA_DIR=${5:-./data}

nsys profile -o "$OUT" \
  python3 examples/pytorch/mnist_cnn.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --data-dir "$DATA_DIR"
