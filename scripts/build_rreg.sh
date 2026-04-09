#!/usr/bin/env bash
set -euo pipefail

RREG=${1:-64}

cmake -S . -B build-rreg \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas -v -maxrregcount=${RREG}"
cmake --build build-rreg -j --verbose | tee "build-rreg_${RREG}.log"
