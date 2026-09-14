#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
cd "$(dirname "$0")"
results=${RESULTS_DIR:-results/development/radix-current}
mkdir -p "$results"
SOURCE=benchmark_radix.cu OUTPUT=benchmark-radix ./build.sh
./benchmark-radix --test > "$results/correctness.txt"
./benchmark-radix --stress > "$results/stress.txt"
for bits in 4 6 7; do
  SOURCE=benchmark_radix.cu OUTPUT="benchmark-radix-bits$bits" \
    NVCC_FLAGS="${NVCC_FLAGS:-} -DLMR_RADIX_BITS=$bits" ./build.sh
  "./benchmark-radix-bits$bits" --test > "$results/correctness-bits$bits.txt"
done
sanitizer=${COMPUTE_SANITIZER:-${CUDA_ENV:+"$CUDA_ENV/bin/"}compute-sanitizer}
for tool in memcheck initcheck racecheck synccheck; do
  log="$results/$tool.txt"
  : > "$log"
  # Tagged records, partial pages, small worker grids, and multiple history-ring wraps.
  for configuration in "262147 3 duplicates" "524291 17 random"; do
    read -r n workers distribution <<< "$configuration"
    "$sanitizer" --tool "$tool" --error-exitcode 2 \
      ./benchmark-radix --check "$n" "$workers" "$distribution" >> "$log" 2>&1
  done
  tail -1 "$log"
done
