#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-nsys_cifar10_resnet}
EPOCHS=${2:-1}
BATCH_SIZE=${3:-256}
MAX_STEPS=${4:-200}
DATA_DIR=${5:-./data}

nsys profile -o "$OUT" \
  python3 examples/pytorch/cifar10_resnet_amp.py \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --max-steps "$MAX_STEPS" \
    --data-dir "$DATA_DIR" \
    --channels-last
