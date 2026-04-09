#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-nsys_baseline}
NUM_ELEMENTS=${2:-33554432}
ITERATIONS=${3:-200}

nsys profile -o "$OUT" ./build-lineinfo/benchmark "$NUM_ELEMENTS" "$ITERATIONS"
