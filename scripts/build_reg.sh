#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas -v"
cmake --build build -j --verbose | tee build-lineinfo.log