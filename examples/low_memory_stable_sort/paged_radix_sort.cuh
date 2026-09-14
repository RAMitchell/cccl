// SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <cub/block/block_radix_rank.cuh>

#include <cuda/std/type_traits>

#include <climits>

#include "paged_stable_sort.cuh"

namespace paged_radix
{
// Experimental stable LSD radix policy for contiguous device ranges and unsigned 32-bit keys.
// The final result occupies the original range; a supplied workspace can be reused.
struct identity_key
{
  template <class T>
  __host__ __device__ T operator()(T x) const
  {
    return x;
  }
};
struct statistics
{
  size_t temporary_bytes = 0;
  int page_items = 0, workers = 0, spare_pages = 0, digit_bits = 0, block_threads = 0;
  float init_ms = 0, histogram_ms = 0, pass_setup_ms = 0, scatter_ms = 0, materialize_ms = 0;
};
template <int Bits = 8, int PageItems = 4096, class Key = identity_key>
struct policy : thrust::execution_policy<policy<Bits, PageItems, Key>>
{
  cudaStream_t stream = nullptr;
  // A worker is a persistent CUDA thread block; zero selects four blocks per SM.
  int workers                    = 0;
  bool profile                   = false;
  paged_sort::workspace* scratch = nullptr;
  statistics* stats              = nullptr;
  Key key{};
  policy on(cudaStream_t s) const
  {
    auto p   = *this;
    p.stream = s;
    return p;
  }
};
namespace detail
{
#ifndef PAGED_RADIX_THREADS
#  define PAGED_RADIX_THREADS 512
#endif
constexpr int threads = PAGED_RADIX_THREADS;
using paged_sort::check;
using paged_sort::detail::free_queue;
using paged_sort::detail::index_t;
using paged_sort::detail::storage;
using paged_sort::detail::ticket_t;
struct digit_extractor
{
  int shift;
  unsigned mask;
  __device__ unsigned Digit(unsigned key) const
  {
    return (key >> shift) & mask;
  }
};

template <class It, int B>
__global__ void prepare_tail(storage<It, B> data, int tail)
{
  for (int i = threadIdx.x; i < tail; i += blockDim.x)
  {
    data.put(data.full_pages, i, data.first[index_t(data.full_pages) * B + i]);
  }
}

// Digit totals are invariant under permutation, so all passes share one initial read.
template <int Bits, class It, class Key>
__global__ void histogram_all(It first, unsigned* totals, index_t n, Key key)
{
  constexpr int bins = 1 << Bits, passes = (32 + Bits - 1) / Bits;
  __shared__ unsigned local[passes][bins];
  for (int i = threadIdx.x; i < passes * bins; i += threads)
  {
    local[i / bins][i % bins] = 0;
  }
  __syncthreads();
  for (index_t i = index_t(blockIdx.x) * threads + threadIdx.x; i < n; i += index_t(gridDim.x) * threads)
  {
    unsigned k = key(first[i]);
#pragma unroll
    for (int pass = 0; pass < passes; ++pass)
    {
      unsigned bin = (k >> (pass * Bits)) & ((1U << Bits) - 1);
      atomicAdd(&local[pass][bin], 1U);
    }
  }
  __syncthreads();
  for (int i = threadIdx.x; i < passes * bins; i += threads)
  {
    atomicAdd(totals + i, local[i / bins][i % bins]);
  }
}
template <int Bits>
__global__ void scan_totals(unsigned* totals)
{
  constexpr int bins = 1 << Bits, passes = (32 + Bits - 1) / Bits;
  if (threadIdx.x < passes)
  {
    unsigned sum = 0;
    for (int i = 0; i < bins; ++i)
    {
      unsigned x                     = totals[threadIdx.x * bins + i];
      totals[threadIdx.x * bins + i] = sum;
      sum += x;
    }
  }
}

__device__ inline int output_page(int* map, unsigned page, free_queue queue)
{
  cuda::atomic_ref<int, cuda::thread_scope_device> entry(map[page]);
  int value = entry.load(cuda::memory_order_acquire);
  if (value == -1)
  {
    int expected = -1;
    if (entry.compare_exchange_strong(expected, -2, cuda::memory_order_acq_rel))
    {
      value = queue.pop();
      entry.store(value, cuda::memory_order_release);
      return value;
    }
    value = expected;
  }
  while (value < 0)
  {
    __nanosleep(32);
    value = entry.load(cuda::memory_order_acquire);
  }
  return value;
}

// Each atomic word contains a tile generation and either a count or an inclusive prefix.
// Padding belongs to the final digit bucket but must not contribute to global offsets.
template <int Bits, int B>
struct publish_counts
{
  ticket_t* history;
  unsigned mask;
  int page, valid;
  unsigned invalid_bin;
  template <int Count>
  __device__ void operator()(int (&counts)[Count]) const
  {
#pragma unroll
    for (int u = 0; u < Count; ++u)
    {
      unsigned bin = threadIdx.x * Count + u;
      if (bin < (1U << Bits))
      {
        unsigned count = counts[u] - (bin == invalid_bin ? B - valid : 0);
        cuda::atomic_ref<ticket_t, cuda::thread_scope_device>(history[(unsigned(page) & mask) * (1U << Bits) + bin])
          .store((ticket_t(page + 1) << 32) | count, cuda::memory_order_release);
      }
    }
  }
};

template <int Bits, int B, class It, class Key>
__global__ void scatter(
  storage<It, B> data,
  const int* input_map,
  int* output_map,
  const unsigned* totals,
  ticket_t* history,
  unsigned* done,
  unsigned history_mask,
  unsigned* control,
  free_queue queue,
  index_t n,
  int np,
  Key key,
  digit_extractor digit)
{
  constexpr int bins = 1 << Bits, items = B / threads;
  using T      = typename storage<It, B>::T;
  using ranker = cub::BlockRadixRankMatchEarlyCounts<threads, Bits, false>;
  __shared__ union
  {
    typename ranker::TempStorage rank;
    T ordered[B];
  } temp;
  __shared__ int starts[bins + 1], destinations[2 * bins];
  __shared__ unsigned positions[bins];
  __shared__ int page;
  while (true)
  {
    if (threadIdx.x == 0)
    {
      // Claim input tiles only when a block is running. A waiting tile therefore never
      // depends on a predecessor owned by an unscheduled block.
      page = int(atomicAdd(control, 1U));
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> frontier(control[1]);
      while (page < np && unsigned(page) - frontier.load(cuda::memory_order_acquire) >= history_mask)
      {
        __nanosleep(32);
      }
    }
    __syncthreads();
    int p = page;
    if (p >= np)
    {
      return;
    }
    int physical = input_map[p];
    int valid    = int(paged_sort::detail::min_(index_t(B), n - index_t(p) * B));
    // CUB Match ranking expects warp-striped input and preserves equal-key order.
    T values[items];
    unsigned keys[items];
#pragma unroll
    for (int u = 0; u < items; ++u)
    {
      int i     = int(threadIdx.x / 32) * (32 * items) + u * 32 + int(threadIdx.x % 32);
      values[u] = i < valid ? data.get(physical, i) : T{};
      keys[u]   = i < valid ? key(values[u]) : UINT_MAX;
    }
    __syncthreads();
    // Every input page has one owner. Its contents now live entirely on chip.
    if (threadIdx.x == 0)
    {
      queue.push(physical);
    }
    int ranks[items], tile_prefix[ranker::BINS_TRACKED_PER_THREAD];
    ranker(temp.rank).RankKeys(
      keys, ranks, digit, tile_prefix, publish_counts<Bits, B>{history, history_mask, p, valid, digit.mask});
    if (threadIdx.x < bins)
    {
      starts[threadIdx.x] = min(tile_prefix[0], valid);
    }
    if (threadIdx.x == 0)
    {
      starts[bins] = valid;
    }
    __syncthreads();
#pragma unroll
    for (int u = 0; u < items; ++u)
    {
      temp.ordered[ranks[u]] = values[u];
    }
    __syncthreads();
    unsigned count  = threadIdx.x < bins ? unsigned(starts[threadIdx.x + 1] - starts[threadIdx.x]) : 0;
    unsigned cursor = 0;
    if (threadIdx.x < bins)
    {
      unsigned sum = count;
      int previous = p - 1;
      while (previous >= 0)
      {
        cuda::atomic_ref<ticket_t, cuda::thread_scope_device> entry(
          history[(unsigned(previous) & history_mask) * bins + threadIdx.x]);
        // The atomic word is self-contained; no other data is acquired through it.
        ticket_t state      = entry.load(cuda::memory_order_relaxed);
        unsigned generation = unsigned(state >> 32), value = unsigned(state);
        if (generation == unsigned(previous + 1))
        {
          sum += value & 0x7fffffffU;
          if (value & 0x80000000U)
          {
            break;
          }
          --previous;
        }
        else if (generation > unsigned(previous + 1))
        {
          // The oldest unfinished tile always retains its predecessor checkpoint.
          // If a farther-back aggregate was recycled, restart from our predecessor.
          sum      = count;
          previous = p - 1;
        }
        else
        {
          __nanosleep(32);
        }
      }
      cursor                 = totals[threadIdx.x] + sum - count;
      positions[threadIdx.x] = cursor;
      cuda::atomic_ref<ticket_t, cuda::thread_scope_device>(history[(unsigned(p) & history_mask) * bins + threadIdx.x])
        .store((ticket_t(p + 1) << 32) | (sum | 0x80000000U), cuda::memory_order_release);
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      // Retire whole epochs to avoid a contended frontier update for every tile.
      // All digit prefixes are published before this block contributes to the epoch.
      unsigned epoch_size   = (history_mask + 1) / 2;
      unsigned epoch        = unsigned(p) / epoch_size;
      unsigned epoch_length = min(epoch_size, unsigned(np) - epoch * epoch_size);
      cuda::atomic_ref<unsigned, cuda::thread_scope_device> completed(done[epoch]);
      if (completed.fetch_add(1, cuda::memory_order_acq_rel) == epoch_length - 1)
      {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> frontier(control[1]);
        unsigned oldest = frontier.load(cuda::memory_order_acquire);
        while (oldest < unsigned(np))
        {
          unsigned length = min(epoch_size, unsigned(np) - oldest);
          if (cuda::atomic_ref<unsigned, cuda::thread_scope_device>(done[oldest / epoch_size])
                .load(cuda::memory_order_acquire)
              != length)
          {
            break;
          }
          unsigned expected = oldest;
          frontier.compare_exchange_strong(expected, oldest + length, cuda::memory_order_acq_rel);
          oldest = frontier.load(cuda::memory_order_acquire);
        }
      }
    }
    if (threadIdx.x < bins && count)
    {
      unsigned first_page = cursor / B, last_page = (cursor + count - 1) / B;
      destinations[threadIdx.x * 2] = output_page(output_map, first_page, queue);
      destinations[threadIdx.x * 2 + 1] =
        last_page == first_page ? destinations[threadIdx.x * 2] : output_page(output_map, last_page, queue);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < valid; i += threads)
    {
      T value              = temp.ordered[i];
      unsigned bin         = digit.Digit(key(value));
      unsigned destination = positions[bin] + unsigned(i - starts[bin]);
      int which            = int(destination / B != positions[bin] / B);
      data.put(destinations[2 * bin + which], int(destination % B), value);
    }
    __syncthreads();
  }
}

template <int Bits, int B, class Key, class It>
void dispatch(policy<Bits, B, Key>& p, It first, index_t n)
{
  using T            = typename thrust::iterator_traits<It>::value_type;
  constexpr int bins = 1 << Bits;
  static_assert(Bits >= 1 && Bits <= 8);
  static_assert(threads >= bins && threads >= 256 && threads <= 1024 && threads % 32 == 0);
  static_assert(B >= threads && B <= 4096 && (B & (B - 1)) == 0 && B % threads == 0);
  static_assert(B * sizeof(T) <= 32768, "page must fit in shared memory");
  int np = int((n + B - 1) / B), full = int(n / B), tail = int(n % B);
  int device, sms;
  check(cudaGetDevice(&device));
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
  int workers = std::min(p.workers ? p.workers : 4 * sms, np);
  // Published prefixes cover one global input prefix: at most two boundaries per bucket.
  int spare         = std::min(2 * bins + 4, np);
  unsigned capacity = 1;
  while (capacity < unsigned(spare + workers + 1))
  {
    capacity *= 2;
  }
  // Keep two epochs and protect the predecessor checkpoint of the oldest
  // unfinished tile. History size depends on the worker grid, not the input size.
  unsigned history_capacity = 1;
  while (history_capacity < unsigned(2 * workers + 2))
  {
    history_capacity *= 2;
  }
  paged_sort::detail::layout l;
  size_t extra_o = l.add<T>(size_t(spare + (tail != 0)) * B);
  size_t map0_o = l.add<int>(np), map1_o = l.add<int>(np);
  constexpr int passes = (32 + Bits - 1) / Bits;
  size_t totals_o = l.add<unsigned>(bins * passes), history_o = l.add<ticket_t>(size_t(history_capacity) * bins),
         done_o    = l.add<unsigned>((np + history_capacity / 2 - 1) / (history_capacity / 2)),
         control_o = l.add<unsigned>(2);
  size_t queue_o = l.add<int>(capacity), seq_o = l.add<ticket_t>(capacity);
  size_t counters_o = l.add<ticket_t>(2), task_o = l.add<unsigned>(1);
  size_t inverse_o = l.add<int>(np + spare), locks_o = l.add<int>(np + spare);
  paged_sort::detail::allocation allocation(l.bytes, p.scratch);
  auto raw = static_cast<unsigned char*>(allocation.ptr);
  auto at  = [&](size_t offset) {
    return raw + offset;
  };
  storage<It, B> data{first, reinterpret_cast<T*>(at(extra_o)), full};
  auto map0 = reinterpret_cast<int*>(at(map0_o)), map1 = reinterpret_cast<int*>(at(map1_o));
  auto totals = reinterpret_cast<unsigned*>(at(totals_o)), done = reinterpret_cast<unsigned*>(at(done_o)),
       control = reinterpret_cast<unsigned*>(at(control_o));
  auto history = reinterpret_cast<ticket_t*>(at(history_o));
  auto inverse = reinterpret_cast<int*>(at(inverse_o)), locks = reinterpret_cast<int*>(at(locks_o));
  auto task = reinterpret_cast<unsigned*>(at(task_o));
  free_queue queue{reinterpret_cast<int*>(at(queue_o)),
                   reinterpret_cast<ticket_t*>(at(seq_o)),
                   reinterpret_cast<ticket_t*>(at(counters_o)),
                   capacity - 1};
  if (p.stats)
  {
    *p.stats = {p.scratch ? p.scratch->capacity : l.bytes, B, workers, spare, Bits, threads};
  }
  using field = float statistics::*;
  std::vector<std::pair<cudaEvent_t, field>> events;
  auto stamp = [&](field f) {
    if (p.profile && p.stats)
    {
      cudaEvent_t event;
      check(cudaEventCreate(&event));
      check(cudaEventRecord(event, p.stream));
      events.push_back({event, f});
    }
  };
  stamp(&statistics::init_ms);
  paged_sort::detail::initialize<<<(std::max(capacity, unsigned(np)) + 255) / 256, 256, 0, p.stream>>>(
    map0, map1, np, spare, queue);
  check(cudaMemsetAsync(data.extra, 0, size_t(spare + (tail != 0)) * B * sizeof(T), p.stream));
  if (tail)
  {
    prepare_tail<It, B><<<1, threads, 0, p.stream>>>(data, tail);
  }
  stamp(&statistics::histogram_ms);
  check(cudaMemsetAsync(totals, 0, size_t(bins * passes) * sizeof(unsigned), p.stream));
  histogram_all<Bits><<<sms * 4, threads, 0, p.stream>>>(first, totals, n, p.key);
  scan_totals<Bits><<<1, 32, 0, p.stream>>>(totals);
  for (int shift = 0; shift < 32; shift += Bits)
  {
    digit_extractor digit{shift, (1U << std::min(Bits, 32 - shift)) - 1};
    stamp(&statistics::pass_setup_ms);
    check(cudaMemsetAsync(map1, 0xff, size_t(np) * sizeof(int), p.stream));
    check(cudaMemsetAsync(history, 0, size_t(history_capacity) * bins * sizeof(ticket_t), p.stream));
    check(cudaMemsetAsync(
      done, 0, size_t((np + history_capacity / 2 - 1) / (history_capacity / 2)) * sizeof(unsigned), p.stream));
    check(cudaMemsetAsync(control, 0, 2 * sizeof(unsigned), p.stream));
    stamp(&statistics::scatter_ms);
    scatter<Bits, B><<<workers, threads, 0, p.stream>>>(
      data,
      map0,
      map1,
      totals + (shift / Bits) * bins,
      history,
      done,
      history_capacity - 1,
      control,
      queue,
      n,
      np,
      p.key,
      digit);
    std::swap(map0, map1);
  }
  stamp(&statistics::materialize_ms);
  check(cudaMemsetAsync(inverse, 0xff, size_t(np + spare) * sizeof(int), p.stream));
  check(cudaMemsetAsync(locks, 0, size_t(np + spare) * sizeof(int), p.stream));
  check(cudaMemsetAsync(task, 0, sizeof(unsigned), p.stream));
  paged_sort::detail::inverse_map<<<(np + 255) / 256, 256, 0, p.stream>>>(map0, inverse, np);
  constexpr int permutation_threads = paged_sort::detail::threads_for<T>;
  paged_sort::detail::materialize<It, B>
    <<<workers, permutation_threads, 0, p.stream>>>(data, map0, inverse, locks, task, np);
  if (tail)
  {
    paged_sort::detail::copy_tail<It, B><<<1, threads, 0, p.stream>>>(data, tail);
  }
  stamp(nullptr);
  check(cudaGetLastError());
  check(cudaStreamSynchronize(p.stream));
  for (size_t i = 1; i < events.size(); ++i)
  {
    float ms;
    check(cudaEventElapsedTime(&ms, events[i - 1].first, events[i].first));
    p.stats->*(events[i - 1].second) += ms;
  }
  for (auto event : events)
  {
    check(cudaEventDestroy(event.first));
  }
}
} // namespace detail

template <int Bits, int B, class Key, class It>
void stable_sort(policy<Bits, B, Key>& p, It first, It last)
{
  using T = typename thrust::iterator_traits<It>::value_type;
  static_assert(thrust::is_contiguous_iterator<It>::value, "contiguous device iterators required");
  static_assert(cuda::std::is_trivially_copyable_v<T> && cuda::std::is_trivially_default_constructible_v<T>);
  static_assert(cuda::std::is_same_v<decltype(p.key(T{})), unsigned>, "key extractor must return unsigned int");
  auto n = last - first;
  if ((cuda::std::is_unsigned_v<decltype(n)> ? false : n < 0) || n > INT_MAX)
  {
    throw std::length_error("radix prototype supports at most INT_MAX records");
  }
  if (p.workers < 0 || p.workers > 65536)
  {
    throw std::invalid_argument("workers must be in [0,65536]");
  }
  if (p.stats)
  {
    *p.stats = {};
  }
  if (n < 2)
  {
    return;
  }
  detail::dispatch(p, first, detail::index_t(n));
}
} // namespace paged_radix
