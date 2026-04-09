#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-ncu_matmul_tiled}
MATRIX_SIZE=${2:-1024}
ITERATIONS=${3:-50}

ncu --set full -o "$OUT" ./build/matmul_tiled_benchmark "$MATRIX_SIZE" "$ITERATIONS"
