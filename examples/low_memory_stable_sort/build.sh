#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set -euo pipefail
cd "$(dirname "$0")"
compiler=${NVCC:-${CUDA_ENV:+"$CUDA_ENV/bin/"}nvcc}
CCCL_ROOT=${CCCL_ROOT:-"$PWD/../.."}
extra=("-I$CCCL_ROOT/thrust" "-I$CCCL_ROOT/cub" "-I$CCCL_ROOT/libcudacxx/include")
# NVCC_FLAGS is intentionally a whitespace-separated list of compiler flags.
read -r -a flags <<< "${NVCC_FLAGS:-}"
"$compiler" -O3 -std=c++17 -arch="${CUDA_ARCH:-sm_120}" -lineinfo -Xcompiler=-Wall \
  "${extra[@]}" "${flags[@]}" "${SOURCE:-benchmark.cu}" -o "${OUTPUT:-benchmark}"
