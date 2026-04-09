#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build-lineinfo \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_FLAGS="-lineinfo -Xptxas -v"
cmake --build build-lineinfo -j --verbose | tee build-lineinfo.log
