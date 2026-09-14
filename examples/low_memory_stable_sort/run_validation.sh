#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
cd "$(dirname "$0")"
results=${RESULTS_DIR:-results/development/current}
mkdir -p "$results"
OUTPUT=benchmark ./build.sh
NVCC_FLAGS="${NVCC_FLAGS:-} -DRECORD_WORDS=4" OUTPUT=benchmark-record16 ./build.sh
SOURCE=correctness_extra.cu OUTPUT=correctness_extra ./build.sh
SOURCE=benchmark_scalar.cu OUTPUT=benchmark-scalar ./build.sh
./benchmark --test > "$results"/correctness-final-8.txt
./benchmark-record16 --test > "$results"/correctness-final-16.txt
./correctness_extra > "$results"/correctness-stress.txt
sanitizer=${COMPUTE_SANITIZER:-${CUDA_ENV:+"$CUDA_ENV/bin/"}compute-sanitizer}
for tool in memcheck initcheck racecheck synccheck; do
  log="$results/$tool-final.txt"
  : > "$log"
  # Exercise both reclamation protocols, including partial pages and queue wrapping.
  for configuration in "benchmark 4096" "benchmark 8192" "benchmark-record16 2048"; do
    read -r binary page <<< "$configuration"
    LMS_WARMUPS=0 "$sanitizer" --tool "$tool" --error-exitcode 2 \
      "./$binary" 262147 "$page" 3 duplicates 1 >> "$log" 2>&1
  done
  tail -1 "$log"
done
