# Stable radix sort with recycled pages

The page-recycling idea carries over to stable LSD radix sort. This experiment uses a
small reserve of physical pages and a bounded history of bucket counts, then puts the
result back into the original array. It uses the same Thrust custom-policy interface as
the merge prototype.

## Results

[One-page headline chart (PDF)](docs/radix_headline.pdf) · [PNG](docs/radix_headline.png)

On an RTX PRO 6000 Blackwell, the prototype retains about **61% of Thrust's radix
throughput on large random 32-bit key arrays**. At 1.07 billion keys it reduces scratch
by **99.6%**, from **4.33 GB to 16.88 MB**, and accounted peak memory from **8.63 GB to
4.31 GB**. This is a large memory saving, but it has not reached the throughput target.

| Keys | Thrust (ms) | Paged radix (ms) | Throughput retained | Scratch MB, Thrust → paged | Peak GB, Thrust → paged |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1.05 M | 0.051 | 0.139 | 36.6% | 4.35 → 6.31 | 0.009 → 0.011 |
| 16.78 M | 0.278 | 0.566 | 49.1% | 69.47 → 12.75 | 0.137 → 0.080 |
| 67.11 M | 1.665 | 2.732 | 60.9% | 277.86 → 12.94 | 0.546 → 0.281 |
| 268.44 M | 7.031 | 11.463 | 61.3% | 1,111.43 → 13.73 | 2.185 → 1.087 |
| 1.07 B | 28.203 | 46.509 | 60.6% | 4,332.68 → 16.88 | 8.628 → 4.312 |

Measured on 14 September 2026 with CUDA 12.9 and this CCCL checkout, which reports
Thrust version 3.6.0. The baseline is the integer radix path of `thrust::stable_sort`.
These are medians of nine trials after five warmups, with alternating algorithm order
and reusable allocations. Timings include final restoration of the original array;
input reset copies and allocation growth are excluded. MB and GB use decimal units.
Peak means the input array plus all algorithm scratch; CUDA context and benchmark
copies are excluded. Both complete outputs are compared after each benchmark.

At 1.05 million keys, the default workspace is larger than Thrust's and the prototype
is substantially slower. At 67.1 million keys, duplicate, equal, sorted, reverse, and
alternating inputs retain approximately 60–65% of Thrust's throughput. The complete
measurements are in [final.csv](results/radix/final.csv). These are fresh radix
measurements, separate from the earlier merge-sort study against CCCL 3.4.2.

Narrower digits trade speed for a smaller reserve and history ring. For 67.1 million
random keys, keeping the same page size, block size, and worker grid:

| Bits per digit | Passes | Scratch (MB) | Paged radix (ms) | Throughput retained |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 8 | 1.13 | 4.470 | 37.2% |
| 6 | 6 | 3.49 | 3.565 | 46.7% |
| 7 | 5 | 6.64 | 3.120 | 53.4% |
| 8 | 4 | 12.94 | 2.740 | 60.7% |

The four-bit variant uses about 1.13 MB of scratch while retaining about 37% of Thrust's
throughput. Select it with `paged_radix::policy<4>`; the default is eight bits.
[Tuning measurements](results/radix/tuning.csv) and a separate
[phase profile](results/radix/profile.txt) are included.

## The algorithm

A conventional GPU radix pass reads one array and scatters into a second array. Writing
directly into the input would overwrite records that another block has not read yet.
Here, logical output positions are mapped to available physical pages. An output page
can occupy any page that has already been consumed, or one of the spare pages.

The default configuration uses 4,096 records per page, eight bits per digit, and four
passes for a 32-bit key. A worker is a persistent CUDA thread block with 512 threads;
the default grid has four blocks per SM. The hardware decides how many blocks are
resident at once.

1. **Count all digits once.** Read the original keys and build histograms for all four
   digits. Prefix sums of these totals give the start of each bucket for each pass.
   Permuting the input does not change the totals, so they are reused across passes.
2. **Load and release an input page.** A running worker claims the next logical input
   page, loads all its records into registers, and returns its physical page to the free
   queue. That page can now be overwritten safely.
3. **Rank locally and find bucket offsets.** CUB's block radix rank primitive groups the
   page's records by the current digit, preserving their order within each bucket.
   The worker publishes its bucket counts early. It obtains each bucket's global offset
   by looking back through preceding pages' counts or completed prefix sums.
4. **Scatter through the output map.** Each bucket's records occupy a contiguous logical
   interval. Workers allocate physical output pages on demand from the free queue.
   Writers sharing a logical output page share its mapping and write disjoint positions.
5. **Repeat with the new map.** The next digit pass reads the logical sequence described
   by the output map. Physical pages can keep moving between input and output roles.
6. **Restore contiguous order once.** After the final digit, reuse the merge prototype's
   page permutation to put the sorted records back into the caller's array.

For example, suppose a page contains `[23a, 12, 23b, 11]`, with the letters marking two
records with the same key. In a decimal units-digit illustration, its local bucket order
is `[11, 12, 23a, 23b]`. Counts from earlier input pages determine where each bucket starts
in the global output. The two `23` records stay in their original order, even when their
output page occupies a completely different physical location. The implementation uses
binary digits, but the reasoning is the same.

Stability follows at two levels: local ranks preserve input order, and global offsets
place earlier input pages before later pages within a bucket. Every digit pass is
stable, so processing digits from least to most significant produces a stable full sort.
Tagged-record tests compare the complete output with `std::stable_sort`; checking scalar
keys alone would not establish stability.

## Why the spare pool stays small

A straightforward design gives every worker a partially filled page for every bucket.
That costs roughly `workers * buckets * page_size` records. This version avoids that
multiplication by allocating pages for global logical output positions.

Consider the farthest input page whose bucket prefix has been computed. Computing that
prefix requires counts from all preceding pages, and counts are published only after
those pages have been loaded and released. The possible output from this consumed input
prefix lies in at most one contiguous interval per bucket. Covering those intervals
with physical pages wastes fewer than two page capacities per bucket at their boundaries.

Consequently, a reserve of `2 * buckets + 4` pages covers the boundary overhead, with
some slack. This bound is independent of the worker count. The implementation caps the
reserve at the total number of input pages for small inputs. A partial final input page
gets one additional padded storage page.

The queue must also accommodate pages released by workers that have not finished
scattering. Its capacity covers the reserve, one such page per worker, and tail slack;
queue slots contain page indices, not records.

## Keeping the prefix history bounded

Saving every page's count for every bucket would itself become a large temporary array.
Instead, workers use a ring of history entries. Each entry is one atomic 64-bit word:
its generation identifies the input page, and its payload holds either a local count or
a completed inclusive prefix.

A worker looks backward from its predecessor, adding counts until it reaches a completed
prefix. This is the decoupled look-back technique used in GPU scans and radix sort.
The implementation publishes local counts during block ranking so successors can begin
before the whole predecessor page has finished scattering. CUB's block histogram barrier
precedes this callback, ensuring the input page has been released before publication.

Ring reuse needs an additional rule. Let `H` be the ring length, a power of two at least
`2 * workers + 2`. Completion is tracked in groups of `H/2` input pages. The oldest group
advances only when every page in it has published every bucket prefix. Admission keeps
the predecessor checkpoint of the oldest unfinished page inside the ring. If a reader
encounters a recycled older aggregate, it restarts from its own predecessor and can
then reach a retained checkpoint.

Updating this frontier once per completed group was an important optimization. Updating
it competitively for every page caused severe atomic contention in the early prototype.
Input pages are claimed dynamically by blocks that are already executing, so a worker
never waits for a predecessor assigned to a block that has not been scheduled.

For `N` records of `S` bytes, `B` records per page, `D` buckets, and `P` workers, the main
scratch terms are:

| Component | Approximate bytes |
| --- | ---: |
| Spare record pages | `(2D + 4) * B * S` |
| Prefix-history ring | `8 * H * D`, with `H = O(P)` |
| Two page maps, inverse map, and permutation locks | `16 * ceil(N/B)` |
| Queue, histogram, completion counters, padding | Smaller additional terms |

Overall this is `O(DBS + PD + N/B)` bytes for fixed key width. The actual measurements
include all these terms. With the implemented fixed page size, the metadata term is
small but still linear in `N`; this implementation does not claim a strictly sublinear
asymptotic bound. A larger page size could balance the terms mathematically, but the
on-chip storage and occupancy constraints matter in practice.

## Use and reproduce

```cpp
#include <thrust/sort.h>
#include "paged_radix_sort.cuh"

paged_sort::workspace scratch;
paged_radix::policy<> policy;
policy.scratch = &scratch;
thrust::stable_sort(policy.on(stream), first, last);  // unsigned 32-bit keys
```

The policy also accepts a key extractor as its third template argument. It must be
callable on the device and return `unsigned int`; records must be trivially copyable
and trivially default-constructible. The sort moves whole records and
preserves order among equal extracted keys. See the tagged-record tests in
[benchmark_radix.cu](benchmark_radix.cu). The default 4,096-record page supports records
up to eight bytes; smaller page configurations can fit larger records within the
32 KiB record-storage limit.

```bash
CUDA_ARCH=sm_120 SOURCE=benchmark_radix.cu OUTPUT=benchmark-radix ./build.sh
# Arguments: keys, timed repetitions, worker blocks (0=auto), distribution
./benchmark-radix 67108864 9 0 random

./run_radix_validation.sh
python run_radix_benchmarks.py --full --tuning
```

Set `CUDA_VISIBLE_DEVICES` to select a GPU. Set `CUDA_ENV`, or explicit `NVCC` and
`COMPUTE_SANITIZER` paths, if the toolchain is not on `PATH`. New results default to
`results/development/radix-current`; `RESULTS_DIR` selects another directory. The saved
[results and validation logs](results/radix/) include compiler and GPU metadata.
`LMS_PROFILE=1` prints phase timings from an additional run after the benchmark.

Validation covers 1,860 exact stable-reference cases across four digit widths, including
empty and tiny inputs, partial pages, guard records, duplicate keys, nondefault streams,
raw pointers, `thrust::device_ptr`, workspace reuse, and long history-ring reuse. Eight
Compute Sanitizer runs cover two worker configurations with each of memcheck, initcheck,
racecheck, and synccheck. All passed; the sanitizer runs reported no errors or warnings.

## What remains

The experiment achieves the memory reduction, but its measured throughput still falls
short of the original goal of approaching Thrust. The scatter passes account for most
of the time, and final physical page permutation adds a substantial extra cost. Larger
pages, larger thread blocks, and narrower digits did not beat the selected defaults on
this GPU. Narrower digits remain useful when an even smaller scratch budget matters.

This is a standalone prototype using helpers from the merge implementation and CUB's
block radix rank primitive. It supports ascending unsigned 32-bit keys or extracted keys,
contiguous device iterators, and at most `INT_MAX` records. It synchronizes the selected
stream. General comparator dispatch, signed/floating-point key encodings, 64-bit keys,
`stable_sort_by_key`, asynchronous policies, and graph capture are not implemented.
Workspaces must not be used concurrently. All measurements and sanitizer checks use one
GPU architecture; further architecture testing and concurrency review are still needed
before treating it as a library-quality implementation.

The upfront all-digit histogram and decoupled look-back draw on
[Onesweep: A Faster Least Significant Digit Radix Sort for GPUs](https://arxiv.org/abs/2206.01784)
and the CUB implementation in this checkout. The experiment combines those techniques
with recycled physical pages and a bounded prefix-history ring. No claim of algorithmic
novelty is made here.
