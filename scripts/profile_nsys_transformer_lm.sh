#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-nsys_transformer_lm}
STEPS=${2:-300}
BATCH_SIZE=${3:-64}
SEQ_LEN=${4:-256}
AMP_MODE=${5:-amp}

AMP_FLAG=()
if [[ "$AMP_MODE" == "noamp" ]]; then
  AMP_FLAG=(--disable-amp)
fi

nsys profile -o "$OUT" \
  python3 examples/pytorch/transformer_lm_amp.py \
    --steps "$STEPS" \
    --batch-size "$BATCH_SIZE" \
    --seq-len "$SEQ_LEN" \
    --d-model 512 \
    --nhead 8 \
    --num-layers 6 \
    --ff-dim 2048 \
    "${AMP_FLAG[@]}"
