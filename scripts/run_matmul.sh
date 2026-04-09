#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-matmul.csv}
MATRIX_SIZE=${2:-1024}
ITERATIONS=${3:-50}

./build/matmul_benchmark "$MATRIX_SIZE" "$ITERATIONS" | tee "$OUT"
