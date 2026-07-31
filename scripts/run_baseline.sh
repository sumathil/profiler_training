#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-baseline.csv}
NUM_ELEMENTS=${2:-33554432}
ITERATIONS=${3:-200}

./build/benchmark "$NUM_ELEMENTS" "$ITERATIONS" | tee "$OUT"
