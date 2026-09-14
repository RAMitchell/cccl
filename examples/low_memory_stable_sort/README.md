# Paged stable sort: GPU prototype

This prototype opts into a stable comparison sort through the existing Thrust interface:

```cpp
#include <thrust/sort.h>
#include "paged_stable_sort.cuh"

paged_sort::workspace scratch;       // optional: retain scratch between calls
paged_sort::statistics stats;
paged_sort::policy policy;
policy.scratch = &scratch;
policy.stats = &stats;

thrust::stable_sort(policy.on(stream), first, last, comparator);
// The complete stable result occupies the original [first, last) range.
// stats.temporary_bytes includes data pages AND every device metadata allocation.
```

The no-comparator overload also works. Omitting `policy.scratch` makes each call allocate and
release its workspace. The custom policy is required; ordinary Thrust calls retain their
original implementation. This is a standalone experimental example; it does not change the built-in Thrust implementation.

The promising case is large comparison sorts of records. On the RTX PRO 6000 Blackwell,
experiments against CCCL 3.4.2 approach the original throughput while reducing scratch by
roughly two orders of magnitude on the largest inputs. Read the measured results, including
slower cases, in [results/report.md](results/report.md). Raw measurements and source hashes
are beside that report.

The [illustrated PDF guide](docs/paged_stable_sort_explained.pdf) explains the algorithm
with a worked example and presents the refined results, including peak memory. Its
[document source and rebuild instructions](docs/README.md) are included.

## Implementation

Runs are sequences of logical pages. A stable merge writes into a free physical page and
recycles input pages when their last reader has loaded their records. Pages that fit in a
shared-memory tile release their inputs before allocating output; larger pages release after
completing all tiles. Intermediate runs keep their page
maps; one parallel page permutation restores contiguous physical order at the end.

The refined version also shares metadata storage between merge and final-permutation phases,
sizes the free queue by the worker reserve, and queries only the needed SM-count attribute
instead of fetching full device properties on every call.

The implementation uses ordinary CUDA kernels, CUB block collectives, device-scope atomics,
and a bounded pool of spare pages. It does not require cooperative launch, a grid-wide
barrier, a second full record array, or unique keys. See [DESIGN.md](DESIGN.md) for the
invariants, progress argument, and exact workspace accounting.

For `P` worker blocks, `B` records per page, `N` records, and record size `S`, device scratch is
`O(S*P*B + N/B)` bytes. Choosing `B` proportional to `sqrt(N/P)` for fixed record size gives
`O(sqrt(N*P))` workspace. The implemented automatic policy balances these terms, rounds up to
a power of two, and caps pages at 65,536 records. A fixed page size, or eventually reaching
that cap, gives a small **linear** metadata term; this is not an unlimited asymptotic
sublinear-space guarantee.

## Build and reproduce

The build defaults to headers in this CCCL checkout and `nvcc` on `PATH`. Set `CUDA_ENV`
to a CUDA toolchain prefix or `NVCC` to a compiler path if needed. CUDA 12.9, C++17, and
`sm_120` were used for the recorded study; set `CUDA_ARCH` for your GPU.

```bash
cd examples/low_memory_stable_sort
export CUDA_ARCH=sm_120
./run_validation.sh
python run_benchmarks.py --full
```

`run_validation.sh` builds all four executables, runs the CPU-reference and stress suites,
and runs Compute Sanitizer. Set `COMPUTE_SANITIZER` if it is not on `PATH` or in `CUDA_ENV`.
New validation and benchmark output goes to `results/development/current`, preserving the
committed study. Set `RESULTS_DIR` to choose another output directory. The benchmark runner
records the actual CCCL commit, compiler, GPU, source hashes, warmups, and repeat count.

For a quick build and one experiment:

```bash
./build.sh
# arguments: records, page size (0=automatic), workers (0=automatic), input, repeats
./benchmark 67108864 0 0 random 9
LMS_PROFILE=1 ./benchmark 67108864 0 0 random 3
```

### Recorded results and provenance

The committed report and PDF describe measurements against **CCCL 3.4.2**
(`d36012203ef73ac7f966e848dd88482273e91e02`) on 13 September 2026. They are historical
measurements, not a benchmark of this branch against current CCCL. This branch starts from
NVIDIA `main` at `903a383421d20d85271f29afc5431f3f1fc5f151`.

The import applies CCCL formatting and license headers to the C++ sources and makes the
build scripts portable. [import-provenance.json](import-provenance.json) records the
original prototype revisions, measured source hashes, and imported source hashes.
[PORTING.md](PORTING.md) records validation against this checkout. The algorithm and tuning
choices are carried over from the refined prototype.

To use the historical dependency, clone it and set `CCCL_ROOT` before building:

```bash
git clone --depth 1 --branch v3.4.2 https://github.com/NVIDIA/cccl.git deps/cccl-3.4.2
export CCCL_ROOT="$PWD/deps/cccl-3.4.2"
./run_validation.sh
python run_benchmarks.py --full
```

`python make_report.py` rebuilds the committed historical report and plot; the PDF has
separate [rebuild instructions](docs/README.md). These document generators use the
committed study in `results/`, not the new output under `results/development/current`.

Input distributions are `random`, `duplicates` (17 keys), `equal`, `sorted`, `reverse`,
`runs` (two separated runs), and `alternating` (two interleaving runs). Record benchmarks
retain original positions and compare the entire output against Thrust. The CPU-reference
suite separately verifies stable order. `benchmark-scalar` reports both an explicit
comparison-sort baseline and Thrust's default integer radix path.

## Current limits

- Contiguous device iterators, including raw pointers and `thrust::device_ptr`, only.
  Records must be default-constructible, trivially copyable, and at most 16 bytes.
  This is not yet a replacement for arbitrary fancy iterators or nontrivial objects.
- The call synchronizes its selected stream before returning. Workspace ownership,
  allocator customization, graph capture, asynchronous policies, and concurrent reuse
  are not production Thrust integration features here.
- Large 8- and 16-byte record sorts are the strongest measured case. Plain 32-bit keys
  remain slower than the comparison baseline and substantially slower than the radix
  baseline. Small/medium arrays can also be slower, and the spare-page reserve can exceed
  the original scratch requirement on small inputs. Tiny single-block sorts need no
  device scratch.
- The memory bound concerns auxiliary storage, not reduced memory traffic: a normal merge
  pass still reads and writes the records, and the final permutation adds traffic.
- Testing was on one GPU architecture. The complete benchmark is included so these results
  can be checked on other devices. `PAGED_SORT_THREADS`, `policy.workers`, and
  `policy.page_items` are experimental tuning controls, not portable optimal settings.
- Integer page indices impose the explicit size limit checked in the header. Comparators
  must satisfy the usual strict weak ordering requirement.
