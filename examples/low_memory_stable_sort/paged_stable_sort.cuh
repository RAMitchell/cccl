// SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Experimental stable GPU merge sort with recycled physical pages.
#pragma once
#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_store.cuh>

#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/memory.h>
#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/atomic>
#include <cuda/std/functional>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace paged_sort
{
inline void check(cudaError_t e)
{
  if (e != cudaSuccess)
  {
    throw std::runtime_error(cudaGetErrorString(e));
  }
}
struct statistics
{
  size_t temporary_bytes = 0;
  int page_items = 0, workers = 0, spare_pages = 0, passes = 0;
  float init_ms = 0, block_ms = 0, partition_ms = 0, merge_ms = 0, materialize_ms = 0;
};
struct workspace
{
  void* data                             = nullptr;
  size_t capacity                        = 0;
  workspace()                            = default;
  workspace(workspace const&)            = delete;
  workspace& operator=(workspace const&) = delete;
  ~workspace()
  {
    if (data)
    {
      cudaFree(data);
    }
  }
  void reserve(size_t bytes)
  {
    if (bytes > capacity)
    {
      if (data)
      {
        check(cudaFree(data));
        data     = nullptr;
        capacity = 0;
      }
      check(cudaMalloc(&data, bytes));
      capacity = bytes;
    }
  }
};
struct policy : thrust::execution_policy<policy>
{
  cudaStream_t stream = nullptr;
  int page_items      = 0; // automatic page size
  int workers         = 0;
  statistics* stats   = nullptr;
  bool profile        = false;
  workspace* scratch  = nullptr;
  policy on(cudaStream_t s) const
  {
    auto p   = *this;
    p.stream = s;
    return p;
  }
};
namespace detail
{
using index_t  = long long;
using ticket_t = unsigned long long;
#ifndef PAGED_SORT_THREADS
#  define PAGED_SORT_THREADS 0
#endif
template <class T>
constexpr int threads_for = PAGED_SORT_THREADS ? PAGED_SORT_THREADS : (sizeof(T) <= 4 ? 512 : 256);
template <class T>
__device__ T min_(T a, T b)
{
  return a < b ? a : b;
}
template <class T>
__device__ T max_(T a, T b)
{
  return a > b ? a : b;
}

struct free_queue
{
  int* pages;
  ticket_t* sequence;
  ticket_t* counters; // head, tail
  unsigned mask;
  __device__ int pop() const
  {
    ticket_t t = atomicAdd(counters, 1ULL);
    unsigned s = unsigned(t) & mask;
    cuda::atomic_ref<ticket_t, cuda::thread_scope_device> seq(sequence[s]);
    while (seq.load(cuda::memory_order_acquire) != t + 1)
    {
      __nanosleep(32);
    }
    int p = pages[s];
    seq.store(t + mask + 1ULL, cuda::memory_order_release);
    return p;
  }
  __device__ void push(int p) const
  {
    ticket_t t = atomicAdd(counters + 1, 1ULL);
    unsigned s = unsigned(t) & mask;
    cuda::atomic_ref<ticket_t, cuda::thread_scope_device> seq(sequence[s]);
    while (seq.load(cuda::memory_order_acquire) != t)
    {
      __nanosleep(32);
    }
    pages[s] = p;
    seq.store(t + 1, cuda::memory_order_release);
  }
};

template <class It, int B>
struct storage
{
  using T = typename thrust::iterator_traits<It>::value_type;
  It first;
  T* extra;
  int full_pages;
  __device__ T* address(int p) const
  {
    return p < full_pages ? thrust::raw_pointer_cast(&first[index_t(p) * B]) : extra + index_t(p - full_pages) * B;
  }
  __device__ T get(int p, int i) const
  {
    return p < full_pages ? T(first[index_t(p) * B + i]) : extra[index_t(p - full_pages) * B + i];
  }
  __device__ void put(int p, int i, T const& x) const
  {
    if (p < full_pages)
    {
      first[index_t(p) * B + i] = x;
    }
    else
    {
      extra[index_t(p - full_pages) * B + i] = x;
    }
  }
};
template <class Storage, int B>
struct run
{
  Storage data;
  const int* map;
  index_t start;
  __device__ auto operator[](index_t i) const
  {
    index_t pos = start + i;
    return data.get(map[pos / B], int(pos % B));
  }
};
template <class T>
struct page_span
{
  const T* first;
  const T* second;
  index_t boundary;
  __device__ T operator[](index_t i) const
  {
    return i < boundary ? first[i] : second[i - boundary];
  }
};
template <class It, int B>
__device__ auto cached_span(storage<It, B> data, const int* map, index_t start, index_t count)
{
  using T = typename storage<It, B>::T;
  // Empty spans are never dereferenced, and may start one past the last page.
  if (!count)
  {
    return page_span<T>{nullptr, nullptr, 0};
  }
  int p = int(start / B), offset = int(start % B);
  const T* first  = data.address(map[p]) + offset;
  const T* second = count > B - offset ? data.address(map[p + 1]) : nullptr;
  return page_span<T>{first, second, B - offset};
}
template <class A, class B, class Comp>
__device__ index_t co_rank(A a, B b, index_t na, index_t nb, index_t d, Comp comp)
{
  index_t lo = max_(index_t(0), d - nb), hi = min_(d, na);
  while (lo < hi)
  {
    index_t m = lo + (hi - lo) / 2;
    if (comp(b[d - 1 - m], a[m]))
    {
      hi = m;
    }
    else
    {
      lo = m + 1;
    }
  }
  return lo;
}
struct partition
{
  index_t a0, a1;
};

static __global__ void initialize(int* map, int* refs, int np, int spare, free_queue q)
{
  int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < np)
  {
    map[i]  = i;
    refs[i] = 0;
  }
  if (unsigned(i) <= q.mask)
  {
    q.sequence[i] = i < spare ? ticket_t(i + 1) : ticket_t(i);
    if (i < spare)
    {
      q.pages[i] = np + i;
    }
  }
  if (i == 0)
  {
    q.counters[0] = 0;
    q.counters[1] = spare;
  }
}

template <int Tile, class It, class Comp>
__global__ void tiny_sort(It first, int n, Comp comp)
{
  using T                    = typename thrust::iterator_traits<It>::value_type;
  constexpr int threads      = threads_for<T>;
  constexpr int BlockThreads = Tile < threads ? Tile : threads;
  constexpr int I            = Tile / BlockThreads;
  using Load                 = cub::BlockLoad<T, BlockThreads, I, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Sort                 = cub::BlockMergeSort<T, BlockThreads, I>;
  using Store                = cub::BlockStore<T, BlockThreads, I, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  __shared__ union
  {
    typename Load::TempStorage load;
    typename Sort::TempStorage sort;
    typename Store::TempStorage store;
  } temp;
  T x[I];
  Load(temp.load).Load(first, x, n, T{});
  __syncthreads();
  Sort(temp.sort).StableSort(x, comp, n, x[0]);
  __syncthreads();
  Store(temp.store).Store(first, x, n);
}

template <class It, int B, class Comp>
__global__ void block_sort(storage<It, B> data, index_t n, Comp comp)
{
  using T               = typename storage<It, B>::T;
  constexpr int threads = threads_for<T>;
  constexpr int Leaf    = B < (sizeof(T) <= 8 ? 4096 : 2048) ? B : (sizeof(T) <= 8 ? 4096 : 2048);
  constexpr int I       = Leaf / threads;
  using Load            = cub::BlockLoad<T, threads, I, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Sort            = cub::BlockMergeSort<T, threads, I>;
  using Store           = cub::BlockStore<T, threads, I, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  __shared__ union
  {
    typename Load::TempStorage load;
    typename Sort::TempStorage sort;
    typename Store::TempStorage store;
    T items[Leaf];
  } s;
  index_t start = index_t(blockIdx.x) * Leaf;
  int p = int(start / B), offset = int(start % B);
  int valid = int(min_(index_t(Leaf), n - start));
  T x[I];
  Load(s.load).Load(data.first + start, x, valid, T{});
  __syncthreads();
  Sort(s.sort).StableSort(x, comp, valid, x[0]);
  __syncthreads();
  // Transpose blocked thread data into coalesced physical-page stores.
  for (int j = 0; j < I; ++j)
  {
    s.items[threadIdx.x * I + j] = x[j];
  }
  __syncthreads();
  for (int j = threadIdx.x; j < Leaf; j += threads)
  {
    data.put(p, offset + j, j < valid ? s.items[j] : T{});
  }
}

template <class It, int B, class Comp>
__global__ void make_partitions(
  storage<It, B> data, const int* map, partition* part, int* refs, index_t n, int np, int width, Comp comp)
{
  int t = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= np)
  {
    return;
  }
  index_t base = (index_t(t) / (2LL * width)) * (2LL * width) * B;
  index_t mid  = min_(n, base + index_t(width) * B);
  index_t end  = min_(n, mid + index_t(width) * B);
  index_t na = mid - base, nb = end - mid;
  index_t d0 = index_t(t) * B - base, d1 = min_(d0 + B, na + nb);
  run<storage<It, B>, B> a{data, map, base}, b{data, map, mid};
  if (nb == 0 || !comp(b[0], a[na - 1]))
  {
    part[t] = {-1, 0};
    return;
  }
  if (nb % B == 0 && comp(b[nb - 1], a[0]))
  {
    part[t] = {-2, 0};
    return;
  }
  index_t a0 = co_rank(a, b, na, nb, d0, comp), a1 = co_rank(a, b, na, nb, d1, comp);
  part[t]           = {a0, a1};
  index_t starts[2] = {base + a0, mid + d0 - a0};
  index_t ends[2]   = {base + a1, mid + d1 - a1};
  for (int k = 0; k < 2; ++k)
  {
    if (starts[k] < ends[k])
    {
      for (index_t p = starts[k] / B; p <= (ends[k] - 1) / B; ++p)
      {
        atomicAdd(refs + p, 1);
      }
    }
  }
}

template <class T>
struct fixed_destination
{
  T* pointer;
  __device__ T* operator()() const
  {
    return pointer;
  }
};

template <class T, int Tile>
struct tile_merger
{
  static constexpr int threads = threads_for<T>;
  static constexpr int I       = Tile / threads;
  using Store                  = cub::BlockStore<T, threads, I, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  struct TempStorage
  {
    union
    {
      T input[Tile];
      typename Store::TempStorage store;
    } temp;
    index_t cuts[2];
  };
  TempStorage& temp;
  template <class A, class C, class Comp>
  __device__ void merge(A a, C b, index_t na, index_t nb, index_t offset, int count, T* out, Comp comp)
  {
    merge_with_destination(a, b, na, nb, offset, count, fixed_destination<T>{out}, comp);
  }
  template <class A, class C, class Destination, class Comp>
  __device__ void merge_with_destination(
    A a, C b, index_t na, index_t nb, index_t offset, int count, Destination destination, Comp comp)
  {
    T* s = temp.temp.input;
    if (threadIdx.x < 2)
    {
      temp.cuts[threadIdx.x] = co_rank(a, b, na, nb, offset + (threadIdx.x ? count : 0), comp);
    }
    __syncthreads();
    int ca = int(temp.cuts[1] - temp.cuts[0]), cb = count - ca;
    for (int j = threadIdx.x; j < count; j += threads)
    {
      s[j] = j < ca ? a[temp.cuts[0] + j] : b[offset - temp.cuts[0] + j - ca];
    }
    __syncthreads();
    // All input records are now on chip; the callback may recycle their physical pages.
    T* out   = destination();
    int diag = min_(int(threadIdx.x) * I, count);
    int ai = int(co_rank(s, s + ca, ca, cb, diag, comp)), bi = diag - ai;
    T result[I];
    T ka{}, kb{};
    if (ai < ca)
    {
      ka = s[ai];
    }
    if (bi < cb)
    {
      kb = s[ca + bi];
    }
#pragma unroll
    for (int j = 0; j < I; ++j)
    {
      result[j] = T{};
      if (diag + j < count)
      {
        bool left = bi >= cb || (ai < ca && !comp(kb, ka));
        result[j] = left ? ka : kb;
        if (left)
        {
          ++ai;
          if (j + 1 < I && ai < ca)
          {
            ka = s[ai];
          }
        }
        else
        {
          ++bi;
          if (j + 1 < I && bi < cb)
          {
            kb = s[ca + bi];
          }
        }
      }
    }
    __syncthreads();
    Store(temp.temp.store).Store(out, result, count);
    __syncthreads();
  }
};

// Merge the initial leaves inside each logical page using one recycled page per worker.
// This decouples page size (and thus metadata size) from the on-chip tile size.
template <class It, int B, class Comp>
__global__ void seed_pages(
  storage<It, B> data,
  const int* __restrict__ in_map,
  int* __restrict__ out_map,
  free_queue q,
  index_t n,
  int np,
  int width,
  Comp comp)
{
  using T               = typename storage<It, B>::T;
  constexpr int threads = threads_for<T>;
  constexpr int Tile    = sizeof(T) <= 8 ? 4096 : 2048;
  __shared__ typename tile_merger<T, Tile>::TempStorage temp;
  __shared__ int spare, src;
  if (threadIdx.x == 0)
  {
    spare = q.pop();
  }
  __syncthreads();
  int first = int(index_t(np) * blockIdx.x / gridDim.x), last = int(index_t(np) * (blockIdx.x + 1) / gridDim.x);
  for (int p = first; p < last; ++p)
  {
    if (threadIdx.x == 0)
    {
      src = in_map[p];
    }
    __syncthreads();
    int valid = int(min_(index_t(B), n - index_t(p) * B));
    for (int base = 0; base < valid; base += 2 * width)
    {
      int na = min_(width, valid - base), nb = min_(width, valid - base - na);
      for (int off = 0; off < na + nb; off += Tile)
      {
        tile_merger<T, Tile>{temp}.merge(
          data.address(src) + base,
          data.address(src) + base + na,
          na,
          nb,
          off,
          min_(Tile, na + nb - off),
          data.address(spare) + base + off,
          comp);
      }
    }
    for (int j = valid + threadIdx.x; j < B; j += threads)
    {
      data.put(spare, j, T{});
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      out_map[p] = spare;
      spare      = src;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    q.push(spare);
  }
}

template <class It, int B, class Comp>
__global__ void merge_pages(
  storage<It, B> data,
  const int* __restrict__ in_map,
  int* __restrict__ out_map,
  const partition* __restrict__ part,
  int* __restrict__ refs,
  free_queue q,
  index_t n,
  int np,
  int width,
  Comp comp)
{
  using T               = typename storage<It, B>::T;
  constexpr int threads = threads_for<T>;
  constexpr int Limit   = sizeof(T) <= 8 ? 4096 : 2048;
  constexpr int Tile    = B < Limit ? B : Limit;
  __shared__ typename tile_merger<T, Tile>::TempStorage temp;
  __shared__ int dest;
  // Only lane zero manages this cache; keep its bookkeeping thread-private.
  int cache[8];
  int cached = 0;
  __syncthreads();
  int first_task = int(index_t(np) * blockIdx.x / gridDim.x);
  int last_task  = int(index_t(np) * (blockIdx.x + 1) / gridDim.x);
  for (int t = first_task; t < last_task; ++t)
  {
    auto cut = part[t];
    if (cut.a0 < 0)
    {
      if (threadIdx.x == 0)
      {
        int source = t;
        if (cut.a0 == -2)
        {
          index_t group = (index_t(t) / (2LL * width)) * (2LL * width);
          int right     = int(min_(index_t(width), index_t(np) - group - width));
          int local     = int(t - group);
          source        = int(group) + (local < right ? width + local : local - right);
        }
        out_map[t] = in_map[source];
      }
      continue;
    }

    index_t base = (index_t(t) / (2LL * width)) * (2LL * width) * B;
    index_t mid  = min_(n, base + index_t(width) * B);
    index_t end  = min_(n, mid + index_t(width) * B);
    index_t d0 = index_t(t) * B - base, d1 = min_(d0 + B, end - base);
    index_t astart = base + cut.a0, bstart = mid + d0 - cut.a0;
    index_t na = cut.a1 - cut.a0, nb = d1 - d0 - na;
    auto a = cached_span(data, in_map, astart, na), b = cached_span(data, in_map, bstart, nb);
    // Only lane zero invokes release; all block reads must finish first.
    auto release = [&] {
      index_t starts[2] = {astart, bstart}, ends[2] = {astart + na, bstart + nb};
      for (int k = 0; k < 2; ++k)
      {
        if (starts[k] < ends[k])
        {
          for (index_t p = starts[k] / B; p <= (ends[k] - 1) / B; ++p)
          {
            cuda::atomic_ref<int, cuda::thread_scope_device> ref(refs[p]);
            if (ref.fetch_sub(1, cuda::memory_order_acq_rel) == 1)
            {
              if (cached < 8)
              {
                cache[cached++] = in_map[p];
              }
              else
              {
                q.push(in_map[p]);
              }
            }
          }
        }
      }
    };
    if constexpr (B == Tile)
    {
      auto destination = [&] {
        if (threadIdx.x == 0)
        {
          release();
          dest       = cached ? cache[--cached] : q.pop();
          out_map[t] = dest;
        }
        __syncthreads();
        return data.address(dest);
      };
      tile_merger<T, Tile>{temp}.merge_with_destination(a, b, na, nb, 0, int(d1 - d0), destination, comp);
    }
    else
    {
      if (threadIdx.x == 0)
      {
        dest       = cached ? cache[--cached] : q.pop();
        out_map[t] = dest;
      }
      __syncthreads();
      for (int offset = 0; offset < d1 - d0; offset += Tile)
      {
        int count = int(min_(index_t(Tile), d1 - d0 - offset));
        tile_merger<T, Tile>{temp}.merge(a, b, na, nb, offset, count, data.address(dest) + offset, comp);
      }
    }
    for (int j = int(d1 - d0) + threadIdx.x; j < B; j += threads)
    {
      data.put(dest, j, T{});
    }
    __syncthreads();
    if (threadIdx.x == 0)
    {
      if constexpr (B > Tile)
      {
        release();
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0)
  {
    while (cached)
    {
      q.push(cache[--cached]);
    }
  }
}

static __global__ void inverse_map(const int* map, int* inverse, int np)
{
  int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < np)
  {
    inverse[map[i]] = i;
  }
}

template <class It, int B>
__global__ void materialize(storage<It, B> data, int* map, int* inverse, int* locks, unsigned* task, int np)
{
  constexpr int threads = threads_for<typename storage<It, B>::T>;
  __shared__ int target, src, other, locked;
  while (true)
  {
    if (threadIdx.x == 0)
    {
      target = int(atomicAdd(task, 1U));
    }
    __syncthreads();
    if (target >= np)
    {
      return;
    }
    while (true)
    {
      if (threadIdx.x == 0)
      {
        src    = atomicAdd(map + target, 0);
        locked = 0;
        if (src != target)
        {
          int lo = min_(target, src), hi = max_(target, src);
          cuda::atomic_ref<int, cuda::thread_scope_device> l(locks[lo]), h(locks[hi]);
          int expected = 0;
          if (l.compare_exchange_strong(expected, 1, cuda::memory_order_acquire))
          {
            expected = 0;
            if (h.compare_exchange_strong(expected, 1, cuda::memory_order_acquire))
            {
              if (atomicAdd(map + target, 0) == src)
              {
                locked = 1;
                other  = inverse[target];
              }
              else
              {
                h.store(0, cuda::memory_order_release);
              }
            }
            if (!locked)
            {
              l.store(0, cuda::memory_order_release);
            }
          }
        }
      }
      __syncthreads();
      bool done     = src == target;
      bool acquired = locked;
      // All warps must snapshot the decision before lane zero can start another
      // attempt/task and overwrite the shared control variables.
      __syncthreads();
      if (done)
      {
        break;
      }
      if (acquired)
      {
        for (int j = threadIdx.x; j < B; j += threads)
        {
          auto x = data.get(src, j);
          if (other >= 0)
          {
            auto y = data.get(target, j);
            data.put(src, j, y);
          }
          data.put(target, j, x);
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
          inverse[target] = target;
          inverse[src]    = other;
          if (other >= 0)
          {
            atomicExch(map + other, src);
          }
          atomicExch(map + target, target);
          cuda::atomic_ref<int, cuda::thread_scope_device>(locks[src]).store(0, cuda::memory_order_release);
          cuda::atomic_ref<int, cuda::thread_scope_device>(locks[target]).store(0, cuda::memory_order_release);
        }
        __syncthreads();
        break;
      }
      __nanosleep(64);
    }
  }
}

template <class It, int B>
__global__ void copy_tail(storage<It, B> data, int tail)
{
  for (int i = threadIdx.x; i < tail; i += blockDim.x)
  {
    data.first[index_t(data.full_pages) * B + i] = data.get(data.full_pages, i);
  }
}

struct allocation
{
  void* ptr = nullptr;
  bool owned;
  explicit allocation(size_t bytes, workspace* scratch)
      : owned(!scratch)
  {
    if (scratch)
    {
      scratch->reserve(bytes);
      ptr = scratch->data;
    }
    else
    {
      check(cudaMalloc(&ptr, bytes));
    }
  }
  ~allocation()
  {
    if (owned)
    {
      cudaFree(ptr);
    }
  }
};
struct layout
{
  size_t bytes = 0;
  template <class T>
  size_t add(size_t count)
  {
    bytes         = (bytes + alignof(T) - 1) / alignof(T) * alignof(T);
    size_t offset = bytes;
    bytes += sizeof(T) * count;
    return offset;
  }
};

template <int B, class It, class Comp>
void dispatch(policy& p, It first, index_t n, Comp comp)
{
  using T               = typename thrust::iterator_traits<It>::value_type;
  constexpr int threads = threads_for<T>;
  static_assert(thrust::is_contiguous_iterator<It>::value, "Prototype requires contiguous device iterators");
  static_assert(std::is_trivially_copyable<T>::value, "Prototype requires trivially copyable records");
  int np = int((n + B - 1) / B), full = int(n / B), tail = int(n % B);
  int dev;
  check(cudaGetDevice(&dev));
  int sms;
  check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  int workers                  = p.workers > 0 ? p.workers : 2 * sms;
  workers                      = std::min(workers, np);
  constexpr int Tile           = sizeof(T) <= 8 ? 4096 : 2048;
  constexpr bool early_release = B <= Tile;
  int spare                    = (early_release ? 2 : 3) * workers + 4;
  unsigned capacity            = 1;
  // Early release can temporarily free one extra page per worker before it obtains output.
  while (capacity < unsigned(spare + (early_release ? workers : 0)))
  {
    capacity *= 2;
  }
  layout l;
  size_t extra_o = l.add<T>(size_t(spare + (tail != 0)) * B);
  size_t map0_o = l.add<int>(np), map1_o = l.add<int>(np);
  // Reader counts/partitions and inverse-map/locks have disjoint kernel lifetimes.
  // Each phase starts at the same suitably aligned offset.
  layout merge_l = l, final_l = l;
  size_t refs_o = merge_l.add<int>(np), part_o = merge_l.add<partition>(np);
  size_t inverse_o = final_l.add<int>(np + spare), locks_o = final_l.add<int>(np + spare);
  l.bytes        = std::max(merge_l.bytes, final_l.bytes);
  size_t queue_o = l.add<int>(capacity);
  size_t seq_o = l.add<ticket_t>(capacity), counters_o = l.add<ticket_t>(2), task_o = l.add<unsigned>(1);
  if (p.stats)
  {
    *p.stats = {l.bytes, B, workers, spare, 0};
  }
  allocation alloc(l.bytes, p.scratch);
  if (p.stats && p.scratch)
  {
    p.stats->temporary_bytes = p.scratch->capacity;
  }
  auto raw = static_cast<unsigned char*>(alloc.ptr);
  auto at  = [&](size_t o) {
    return raw + o;
  };
  storage<It, B> data{first, reinterpret_cast<T*>(at(extra_o)), full};
  int* map0 = reinterpret_cast<int*>(at(map0_o));
  int* map1 = reinterpret_cast<int*>(at(map1_o));
  auto refs = reinterpret_cast<int*>(at(refs_o));
  auto part = reinterpret_cast<partition*>(at(part_o));
  auto task = reinterpret_cast<unsigned*>(at(task_o));
  free_queue q{reinterpret_cast<int*>(at(queue_o)),
               reinterpret_cast<ticket_t*>(at(seq_o)),
               reinterpret_cast<ticket_t*>(at(counters_o)),
               capacity - 1};
  using field = float statistics::*;
  std::vector<std::pair<cudaEvent_t, field>> events;
  auto stamp = [&](field f) {
    if (p.profile && p.stats)
    {
      cudaEvent_t e;
      check(cudaEventCreate(&e));
      check(cudaEventRecord(e, p.stream));
      events.push_back({e, f});
    }
  };
  stamp(&statistics::init_ms);
  initialize<<<(std::max(capacity, unsigned(np)) + 255) / 256, 256, 0, p.stream>>>(map0, refs, np, spare, q);
  stamp(&statistics::block_ms);
  constexpr int Leaf = B < (sizeof(T) <= 8 ? 4096 : 2048) ? B : (sizeof(T) <= 8 ? 4096 : 2048);
  if (tail)
  {
    check(cudaMemsetAsync(data.extra, 0, size_t(B) * sizeof(T), p.stream));
  }
  block_sort<It, B><<<int((n + Leaf - 1) / Leaf), threads, 0, p.stream>>>(data, n, comp);
  for (int width = Leaf; width < B; width *= 2)
  {
    seed_pages<It, B><<<workers, threads, 0, p.stream>>>(data, map0, map1, q, n, np, width, comp);
    std::swap(map0, map1);
  }
  for (index_t width = 1; width < np; width *= 2)
  {
    stamp(&statistics::partition_ms);
    make_partitions<It, B><<<(np + 127) / 128, 128, 0, p.stream>>>(data, map0, part, refs, n, np, int(width), comp);
    stamp(&statistics::merge_ms);
    merge_pages<It, B><<<workers, threads, 0, p.stream>>>(data, map0, map1, part, refs, q, n, np, int(width), comp);
    std::swap(map0, map1);
    if (p.stats)
    {
      ++p.stats->passes;
    }
  }
  stamp(&statistics::materialize_ms);
  if (np > 1 || B > Leaf)
  {
    auto inverse = reinterpret_cast<int*>(at(inverse_o)), locks = reinterpret_cast<int*>(at(locks_o));
    check(cudaMemsetAsync(inverse, 0xff, size_t(np + spare) * sizeof(int), p.stream));
    check(cudaMemsetAsync(locks, 0, size_t(np + spare) * sizeof(int), p.stream));
    check(cudaMemsetAsync(task, 0, sizeof(unsigned), p.stream));
    inverse_map<<<(np + 255) / 256, 256, 0, p.stream>>>(map0, inverse, np);
    materialize<It, B><<<workers, threads, 0, p.stream>>>(data, map0, inverse, locks, task, np);
  }
  if (tail)
  {
    copy_tail<It, B><<<1, 256, 0, p.stream>>>(data, tail);
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
  for (auto e : events)
  {
    cudaEventDestroy(e.first);
  }
}
} // namespace detail

template <class It, class Comp>
void stable_sort(policy& p, It first, It last, Comp comp)
{
  if (p.stats)
  {
    *p.stats = {};
  }
  if (first == last)
  {
    return;
  }
  auto n = last - first;
  if (n < 0)
  {
    throw std::invalid_argument("reversed iterator range");
  }
  if (n < 2)
  {
    return;
  }
  if (p.workers < 0 || p.workers > 65536)
  {
    throw std::invalid_argument("workers must be in [0,65536]");
  }
  if ((static_cast<unsigned long long>(n) + 1023) / 1024 > 0x3fffffffULL)
  {
    throw std::length_error("prototype page-index limit exceeded");
  }
  using T                      = typename thrust::iterator_traits<It>::value_type;
  constexpr int launch_threads = detail::threads_for<T>;
  static_assert(sizeof(T) <= 16, "This prototype supports records up to 16 bytes");
  static_assert(std::is_trivially_copyable<T>::value && std::is_default_constructible<T>::value,
                "Prototype requires trivially copyable, default-constructible records");
  static_assert(thrust::is_contiguous_iterator<It>::value, "Prototype requires contiguous device iterators");
  if (n <= (sizeof(T) <= 8 ? 4096 : 2048))
  {
    int tile = n <= 256 ? 256 : n <= 512 ? 512 : n <= 1024 ? 1024 : n <= 2048 ? 2048 : 4096;
    switch (tile)
    {
      case 256:
        detail::tiny_sort<256><<<1, (launch_threads < 256 ? launch_threads : 256), 0, p.stream>>>(first, int(n), comp);
        break;
      case 512:
        detail::tiny_sort<512><<<1, launch_threads, 0, p.stream>>>(first, int(n), comp);
        break;
      case 1024:
        detail::tiny_sort<1024><<<1, launch_threads, 0, p.stream>>>(first, int(n), comp);
        break;
      case 2048:
        detail::tiny_sort<2048><<<1, launch_threads, 0, p.stream>>>(first, int(n), comp);
        break;
      case 4096:
        if constexpr (sizeof(T) <= 8)
        {
          detail::tiny_sort<4096><<<1, launch_threads, 0, p.stream>>>(first, int(n), comp);
        }
        break;
    }
    check(cudaGetLastError());
    check(cudaStreamSynchronize(p.stream));
    if (p.stats)
    {
      p.stats->page_items = tile;
      p.stats->workers    = 1;
    }
    return;
  }
  int page = p.page_items;
  if (page == 0)
  {
    int dev;
    check(cudaGetDevice(&dev));
    int sms;
    check(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    int workers = p.workers > 0 ? p.workers : 2 * sms;
    using T     = typename thrust::iterator_traits<It>::value_type;
    page        = sizeof(T) <= 8 ? 4096 : 2048;
    // Balance page-map bytes against the resident workers' spare pages.
    while (double(page) * page < 48.0 * double(n) / (3.0 * workers * sizeof(T)) && page < 65536)
    {
      page *= 2;
    }
  }
  switch (page)
  {
    case 1024:
      detail::dispatch<1024>(p, first, n, comp);
      break;
    case 2048:
      detail::dispatch<2048>(p, first, n, comp);
      break;
    case 4096:
      detail::dispatch<4096>(p, first, n, comp);
      break;
    case 8192:
      detail::dispatch<8192>(p, first, n, comp);
      break;
    case 16384:
      detail::dispatch<16384>(p, first, n, comp);
      break;
    case 32768:
      detail::dispatch<32768>(p, first, n, comp);
      break;
    case 65536:
      detail::dispatch<65536>(p, first, n, comp);
      break;
    default:
      throw std::invalid_argument("page_items must be zero or a power of two from 1024 to 65536");
  }
}
template <class It>
void stable_sort(policy& p, It first, It last)
{
  using T = typename thrust::iterator_traits<It>::value_type;
  paged_sort::stable_sort(p, first, last, cuda::std::less<T>{});
}
} // namespace paged_sort
