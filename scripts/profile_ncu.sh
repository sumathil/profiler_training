#!/usr/bin/env bash
set -euo pipefail

OUT=${1:-ncu_baseline}
NUM_ELEMENTS=${2:-33554432}
ITERATIONS=${3:-200}

ncu --set full -o "$OUT" ./build-lineinfo/benchmark "$NUM_ELEMENTS" "$ITERATIONS"
