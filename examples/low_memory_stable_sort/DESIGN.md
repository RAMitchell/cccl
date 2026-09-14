# Algorithm and correctness notes

Let `N` be the number of records, `S` their byte size, `B` the number of records in one logical
page, and `P` the number of worker blocks. A logical page stores a contiguous subsequence of
a run. The page map records its physical location, which can be inside the original array
or in the small external reserve.

## Initial sorted pages

CUB's stable block merge sort first sorts leaves that fit in shared memory. If a logical
page is larger than a leaf, `seed_pages` merges those leaves within the page using one spare
page per worker. Logical page size can therefore grow without growing shared-memory use.
A partial last page lives in the external allocation until the final copy back; kernels
never write padding past the user's iterator range.

Single-block inputs take a separate path with no device allocation.

## One merge level

1. `make_partitions` computes stable Merge Path cuts at output page boundaries. It also
   counts how many output tasks read each logical input page. The input map is immutable
   for the entire level.
2. Each worker receives a contiguous interval of output tasks and processes it in order.
   A task reads at most one page of records in total from its two input runs; each span can
   straddle a physical page boundary.
3. When the complete task fits in one shared-memory tile, the worker first loads all its
   input records and synchronizes the block. Lane zero then decrements the input reader
   counts, releases pages whose last reader has finished, and obtains an output page from
   its local cache or the global queue. The output can reuse one of those just-released
   input pages: all subsequent comparisons and merging read the on-chip copy.
4. Larger logical pages use several shared-memory tiles. They obtain a destination before
   starting and release input pages only after the entire page has been merged. This keeps
   later tiles' inputs intact. Both paths publish releases with device-scope acquire/release
   atomics and use the same sequence-numbered free queue.
5. Each output logical page records its chosen physical destination. The next kernel level
   swaps the input/output map roles, not full arrays of records.

A small thread-private cache of released page IDs avoids a queue operation for every page.
Workers flush it before exiting. The global queue uses per-slot sequence numbers and
64-bit enqueue/dequeue tickets to distinguish different uses of a ring slot.

No task overwrites records that another task still needs to read from global memory: it
only writes a physical page that has been released by all readers of the previous contents. No other task sees partial output, because output
runs are consumed in the next kernel level on the same CUDA stream.

## Stability

The initial block sort is stable. In subsequent merges, the left run wins comparator ties.
The co-rank search uses the matching tie convention, including ties that cross task and
page boundaries. Physical page movement does not change the logical sequence.

An already ordered pair of runs keeps its maps. Two strictly separated runs in reverse
order can exchange their page-map segments when the boundary is page aligned. The test
is strict: allowing equal boundary keys here would reverse their stable order. Partial
right pages use the regular merge path instead.

## Why a bounded reserve permits progress

A worker's completed prefix consumes contiguous spans from the input runs. Page-aligned
output boundaries imply that at most two pages' worth of consumed data per worker can be
trapped in partially consumed input pages. Complete intervening merge pairs introduce no
additional loss. Combining spans from different workers can only free more pages.

For pages that fit in one tile (up to 4,096 records for records of at most 8 bytes, or 2,048
records for larger records), set `R = 2P + 4`. Let `L` count tasks that have loaded their inputs
and finished decrementing their reader counts, and `D` count tasks that have acquired their
destination. The early-release ordering guarantees `D <= L`. At least `L - 2P` input pages
have been fully consumed, giving the following lower bound on free physical pages:

```
R + (L - 2P) - D >= R - 2P = 4.
```

For larger pages, retain `R = 3P + 4`. If `C` output pages have completed, at least `C - 2P`
input pages have been fully consumed, and at most `P` additional destinations are in flight:

```
R + (C - 2P) - (C + P) = R - 3P = 4.
```

Some of those pages may be in worker caches rather than the global queue. A worker holding
cached pages can continue, and exiting workers flush their caches. Thus an empty global
queue does not imply that all workers are blocked. A task that already owns a destination
has no dependency on another task's output: its input belongs to the preceding completed
kernel level. This argument assumes eventual scheduling/progress of running CUDA work and
atomic operations; it is not a stronger scheduler fairness guarantee.

The spare count is based on launched workers, even if resource limits prevent all of them
from residing at once. It does not rely on a cooperative launch or on blocks with a
particular index already being resident.

## Final physical permutation

After the final merge, `map[i]` identifies the page containing output page `i`. An inverse
map identifies the logical owner of every occupied physical page.

A worker chooses an output target and locks its target/source physical pages in increasing
page-ID order. It rechecks the source after locking, exchanges page contents, and updates
both maps. Lock acquisition releases a partial acquisition immediately if the second lock
cannot be acquired. Every successful transaction fixes a target; a fixed target is never
subsequently displaced. Free destinations need only one page copy.

A block barrier separates snapshots of the shared decision variables from their next
update. This matters when one warp skips an already fixed target or retries a lock before
other warps have read the decision. Racecheck exposed that issue during development; the
final validation logs refer to the corrected implementation.

At most `Q = ceil(N/B)` successful target-fixing transactions are needed. Each transaction
moves at most two pages, so record traffic is `O(NS)`. This avoids a serial walk through a
potentially enormous permutation cycle. The final partial page is copied to the original
range using only its valid record count.

## Complete workspace accounting

Let `delta` be one for a partial last page and zero otherwise. Let `E` be one for early
release and zero for larger pages, so `R = (3-E)P + 4`.

The free queue needs capacity for at most `R + EP = 3P + 4` pages. In the larger-page path,
freeing input cannot get ahead of allocating output. Early release can temporarily put one
additional page per worker into the pool before the worker obtains a destination. Reader
counts prevent releasing a page more than once. The partial page does not increase this
bound: the number of completely consumed input pages cannot exceed the number of output
page tasks whose input records have been consumed. Set `K` to the smallest power of two
at least `3P + 4`; it no longer depends on `Q`.

Reader counts and partitions are needed during merging. Inverse ownership and locks are
needed only during the final permutation. These two phases share the same allocation
region, with a kernel boundary separating their lifetimes.

| Allocation | Bytes |
|---|---:|
| Spare records and partial page | `(R + delta) * B * S` |
| Two page maps | `8Q` |
| Shared phase region | `max(20Q + 4*(Q mod 2), 8*(Q + R))` |
| Free queue page IDs and sequence numbers | `12K` |
| Queue tickets and final task counter | `20` |

The merge phase region contains `4Q` reader-count bytes, up to four alignment bytes, and
`16Q` partition bytes. The final phase region contains `4(Q+R)` bytes each for ownership
and locks. The resulting exact allocation size is:

```
(R + delta)BS + 8Q + max(20Q + 4*(Q mod 2), 8*(Q + R)) + 12K + 20
```

This is `O(SPB + N/B)` bytes. Shared memory and registers are bounded by the active kernel
tiles and worker count; there is no hidden global array of per-record indices or payloads.
Optional host-side timing events use `O(log N)` bookkeeping. A reused workspace may retain
a larger previous allocation; its capacity is reported rather than silently counting only
the current request.

The automatic page policy rounds a balance of the data-reserve and metadata terms to a
power of two, with a practical minimum tile size and a maximum of 65,536 records. For fixed
`P` and `S`, growing `B` in proportion to `sqrt(N/P)` gives the desired square-root workspace
behavior before that cap. Fixed `B` yields `O(N/B)` metadata; the measured finite-range
space reductions must not be described as an unqualified asymptotic guarantee.

## Related work and scope of the experiment

- [CCCL 3.4.2 merge-sort dispatch](https://github.com/NVIDIA/cccl/blob/v3.4.2/cub/cub/device/dispatch/dispatch_merge_sort.cuh)
  is the full-buffer implementation used by the comparison baseline.
- [Virtual-Memory Powersort](https://arxiv.org/abs/2605.27147) reuses consumed input pages in a
  stable CPU mergesort. This prototype explores a GPU schedule, reclamation protocol,
  and parallel materialization; it does not claim the page-recycling idea is new.
- [On the improvement of the in-place merge algorithm parallelization](https://arxiv.org/abs/2005.12648)
  is relevant to the alternative strategy of partitioning/rearranging input into independent
  in-place merges. Repeated rotations were not selected for this prototype.

The experiments establish a working, promising variant and identify its current performance
limits. They do not establish novelty, production readiness, or a hardware-independent
throughput guarantee.
