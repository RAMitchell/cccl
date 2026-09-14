// SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_ptr.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "paged_stable_sort.cuh"
struct item
{
  unsigned key, sequence;
};
struct less_key
{
  __host__ __device__ bool operator()(item a, item b) const
  {
    return a.key < b.key;
  }
};
int main()
try
{
  std::mt19937 rng(731);
  int cases = 0;
  cudaStream_t stream;
  paged_sort::check(cudaStreamCreate(&stream));
  paged_sort::workspace scratch;
  // Guarded subranges, non-power-of-two lengths, and awkward worker counts.
  for (int workers : {1, 3, 17, 188, 376, 752, 1024})
  {
    for (int page : {1024, 4096, 16384})
    {
      for (int distribution = 0; distribution < 3; ++distribution)
      {
        size_t n = (1 << 20) + rng() % 8193;
        std::vector<item> x(n + 6);
        for (size_t i = 0; i < x.size(); ++i)
        {
          x[i] = {0xdeadbeef, 0xbaadf00d};
        }
        for (size_t i = 0; i < n; ++i)
        {
          unsigned k = distribution == 0 ? rng()
                     : distribution == 1 ? rng() % 5
                                         : (i < n / 2 ? unsigned(i) * 2 : unsigned(n / 2));
          x[i + 3]   = {k, unsigned(i)};
        }
        auto want = x;
        std::stable_sort(want.begin() + 3, want.end() - 3, less_key{});
        item* d;
        paged_sort::check(cudaMalloc(&d, x.size() * sizeof(item)));
        paged_sort::check(cudaMemcpyAsync(d, x.data(), x.size() * sizeof(item), cudaMemcpyHostToDevice, stream));
        paged_sort::policy p;
        p.workers    = workers;
        p.page_items = page;
        p.stream     = stream;
        p.scratch    = &scratch;
        thrust::stable_sort(p, thrust::device_pointer_cast(d) + 3, thrust::device_pointer_cast(d) + 3 + n, less_key{});
        paged_sort::check(cudaMemcpy(x.data(), d, x.size() * sizeof(item), cudaMemcpyDeviceToHost));
        paged_sort::check(cudaFree(d));
        if (std::memcmp(x.data(), want.data(), x.size() * sizeof(item)))
        {
          throw std::runtime_error("guarded stress mismatch");
        }
        ++cases;
      }
    }
  }
  // The no-comparator overload, including signed zeros whose stable order is observable in bits.
  for (int n : {0, 1, 2, 31, 256, 513, 1025, 2049, 4097, 16385, 65539})
  {
    std::vector<double> x(n);
    for (int i = 0; i < n; ++i)
    {
      x[i] = i % 3 == 0 ? -0.0 : i % 3 == 1 ? 0.0 : double(int(rng() % 31) - 15);
    }
    auto want = x;
    std::stable_sort(want.begin(), want.end());
    double* d;
    paged_sort::check(cudaMalloc(&d, std::max(n, 1) * sizeof(double)));
    paged_sort::check(cudaMemcpyAsync(d, x.data(), n * sizeof(double), cudaMemcpyHostToDevice, stream));
    paged_sort::policy p;
    p.stream  = stream;
    p.scratch = &scratch;
    thrust::stable_sort(p, d, d + n);
    paged_sort::check(cudaMemcpy(x.data(), d, n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d);
    if (n && std::memcmp(x.data(), want.data(), n * sizeof(double)))
    {
      throw std::runtime_error("double stability mismatch");
    }
    ++cases;
  }
  // Empty ranges allocate nothing, including null empty ranges.
  paged_sort::statistics stats;
  paged_sort::policy p;
  p.stats = &stats;
  item* d = nullptr;
  thrust::stable_sort(p, d, d, less_key{});
  if (stats.temporary_bytes)
  {
    throw std::runtime_error("empty allocation");
  }
  cudaStreamDestroy(stream);
  std::cout << "PASS " << cases << " guarded/concurrency/default-comparator stress cases\n";
}
catch (std::exception const& e)
{
  std::cerr << e.what() << '\n';
  return 1;
}
