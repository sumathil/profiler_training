#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-nsys_matmul_tiled}
MATRIX_SIZE=${2:-1024}
ITERATIONS=${3:-50}

nsys profile -o "$OUT" ./build/matmul_tiled_benchmark "$MATRIX_SIZE" "$ITERATIONS"
