# CCCL fork import and validation

This example carries the refined paged stable sort prototype into CCCL as a standalone
experiment. The branch is based on NVIDIA `main` at
`903a383421d20d85271f29afc5431f3f1fc5f151`; that checkout reports `THRUST_VERSION=300600`.

## What changed during import

- Added Apache-2.0 WITH LLVM-exception SPDX banners and applied CCCL's pinned
  clang-format 22.1.5. Each of the four CUDA source files was checked against the
  measured source after those same transformations; they match exactly.
- Made build scripts use the enclosing CCCL checkout and discover `nvcc` through
  `PATH`, `NVCC`, or `CUDA_ENV`. An explicit `CCCL_ROOT` still selects another checkout.
- Defaulted new validation and benchmark output to `results/development/current`.
- Updated documentation paths and regenerated the illustrated PDF. PDF text was
  compared with the original: only the example path and quote typography changed.

The original CSV files and environment metadata are preserved byte-for-byte. Their
measurements used CCCL 3.4.2, not this newer checkout. The report and PDF retain that
version and the measured implementation revision. `import-provenance.json` maps the
original source hashes to the formatted import. The original prototype revisions were
local XGBoost worktree commits; this branch does not import XGBoost history.

## Validation on 14 September 2026

CUDA 12.9.86, `sm_120`, one RTX PRO 6000 Blackwell Workstation Edition with 188 SMs:

| Check | Result |
|---|---:|
| Exact CPU stable-sort comparisons, 8-byte records | 1,792 passed |
| Exact CPU stable-sort comparisons, 16-byte records | 1,792 passed |
| Guarded ranges, worker counts, stream, default-comparator stress | 74 passed |
| Compute Sanitizer memcheck | 3 configurations; 0 errors |
| Compute Sanitizer initcheck | 3 configurations; 0 errors |
| Compute Sanitizer racecheck | 3 configurations; 0 hazards |
| Compute Sanitizer synccheck | 3 configurations; 0 errors |
| Small benchmark smoke runs, records and scalar keys | 3 passed |
| CCCL pre-commit checks on imported files | Passed |

Validation used `CUDA_ENV=/home/rorym/miniforge3/envs/xgboost ./run_validation.sh`.
See [the current-CCCL logs and metadata](results/cccl-main-validation/) for the exact
commands, source hashes, and outputs. The three smoke runs check complete results
against Thrust. They do not repeat the full performance study.

For another toolchain, set `NVCC` and `COMPUTE_SANITIZER`, or put both on `PATH`, then
run `./run_validation.sh`. Set `CUDA_ARCH` for the target GPU.
