# Paged stable sort

An experimental stable GPU comparison sort that recycles input pages as a merge consumes
them. It returns the complete result in the original array, using a small pool of spare
pages instead of a second full array.

For 1.07 billion random 8-byte records on an RTX PRO 6000 Blackwell, the prototype uses
**32.14 MB of scratch and retains 86.2% of Thrust's comparison-sort throughput**. Accounted
peak memory falls from 17.18 GB to 8.62 GB. These measurements use CCCL 3.4.2 and CUDA 12.9;
peak memory includes the input and scratch, excluding CUDA context and benchmark overhead.

- [Illustrated PDF](docs/paged_stable_sort_explained.pdf): an approachable explanation with a worked merge.
- [Results](results/report.md): timings, scratch and peak memory, refinements, and slower cases.
- [Algorithm details](DESIGN.md): scheduling, stability, progress, and workspace bounds.
- [Radix experiment](RADIX.md): a stable integer variant with bounded page recycling, timings, and peak memory.

## Use through Thrust

```cpp
#include <thrust/sort.h>
#include "paged_stable_sort.cuh"

paged_sort::workspace scratch;  // retain workspace between calls
paged_sort::policy policy;
policy.scratch = &scratch;

thrust::stable_sort(policy.on(stream), first, last, comparator);
```

The custom policy selects the prototype. The call synchronizes its selected stream and
returns the stable result in `[first, last)`. The no-comparator overload also works.
Omitting `policy.scratch` makes each call allocate and release its workspace. To inspect
scratch usage, point `policy.stats` at a `paged_sort::statistics` object; its
`temporary_bytes` includes both data pages and metadata.

## How it works

Sorted runs are sequences of logical pages, with a map to their physical locations.
Merge Path partitions a merge into output pages. A worker loads its input and releases
each input page after its last reader has finished loading it. Freed pages become storage
for later output. One final page permutation restores contiguous physical order.

A worker is a CUDA thread block. For `P` workers, `B` records per page, `N` records, and
record size `S`, auxiliary storage is `O(S*P*B + N/B)` bytes. The first term is the spare
page pool; the second is metadata. For fixed record size, choosing `B` proportional to
`sqrt(N/P)` gives `O(sqrt(N*P))` storage. The implemented page-size cap of 65,536 records
eventually leaves a small linear metadata term.

## Build and run

Put `nvcc` on `PATH` and set `CUDA_ARCH` for your GPU. The build uses C++17 and headers
from this CCCL checkout.

```bash
cd examples/low_memory_stable_sort
CUDA_ARCH=sm_120 ./build.sh
# arguments: records, page size (0=auto), workers (0=auto), input, repeats
./benchmark 67108864 0 0 random 9
```

For all correctness checks, Compute Sanitizer checks, and the full benchmark:

```bash
export CUDA_ARCH=sm_120
./run_validation.sh
python run_benchmarks.py --full
```

`run_validation.sh` builds all four executables. Set `NVCC` and `COMPUTE_SANITIZER` to
explicit paths, or `CUDA_ENV` to their toolchain prefix, if needed. New results go to
`results/development/current`; `RESULTS_DIR` selects another output directory.

The published study in `results/` was measured on 13 September 2026 against CCCL 3.4.2.
To use the same dependency, set `CCCL_ROOT` to a CCCL 3.4.2 checkout before building.
The [validation logs for this branch](results/cccl-main-validation/) also record successful
runs of 3,658 correctness and stress cases, 12 sanitizer configurations, and three smoke
benchmarks against the newer CCCL checkout. Those smoke runs do not replace the full study.

Input distributions are `random`, `duplicates` (17 keys), `equal`, `sorted`, `reverse`,
`runs` (two separated runs), and `alternating` (two interleaving runs). Record benchmarks
compare the entire output against Thrust, including original positions and payloads.
`benchmark-scalar` compares plain 32-bit keys with both comparison and radix baselines.
Set `LMS_PROFILE=1` for per-phase timings.

`python make_report.py` regenerates the saved study's report and plot (requires matplotlib).
The PDF has separate [rebuild instructions](docs/README.md).

## Current limits

- Contiguous device iterators only, including raw pointers and `thrust::device_ptr`.
  Records must be default-constructible, trivially copyable, and at most 16 bytes.
- Large 8- and 16-byte record sorts are the strongest measured case. Plain 32-bit keys
  are slower than both baselines; small inputs can also be slower and use more scratch.
- The implementation synchronizes the selected stream. Graph capture, asynchronous
  policies, allocator customization, and concurrent workspace reuse are not supported.
- Normal merge passes read and write the records, and the final permutation adds traffic.
  Testing covers one GPU architecture; tuning choices are experimental.
- Comparators must provide a strict weak ordering. Integer page indices impose the
  explicit size limit checked in the header.
