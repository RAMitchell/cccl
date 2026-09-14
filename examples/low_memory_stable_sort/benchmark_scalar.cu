// SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/sort.h>
#include <thrust/version.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

#include "paged_stable_sort.cuh"
struct numeric_less
{
  __host__ __device__ bool operator()(unsigned a, unsigned b) const
  {
    return a < b;
  }
};
struct allocator
{
  using value_type = char;
  paged_sort::workspace* cache;
  char* allocate(std::ptrdiff_t n)
  {
    cache->reserve(n);
    return static_cast<char*>(cache->data);
  }
  void deallocate(char*, size_t) {}
};
template <class F>
float time_it(F f)
{
  cudaEvent_t a, b;
  paged_sort::check(cudaEventCreate(&a));
  paged_sort::check(cudaEventCreate(&b));
  cudaEventRecord(a);
  f();
  cudaEventRecord(b);
  paged_sort::check(cudaEventSynchronize(b));
  float ms;
  cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms;
}
int main(int argc, char** argv)
try
{
  size_t n    = argc > 1 ? std::stoull(argv[1]) : size_t(1) << 26;
  int repeats = argc > 2 ? std::stoi(argv[2]) : 7;
  if (!n || repeats < 1)
  {
    throw std::invalid_argument("positive n and repeats required");
  }
  std::mt19937 rng(42);
  std::vector<unsigned> x(n);
  for (auto& a : x)
  {
    a = rng();
  }
  unsigned *d, *original;
  paged_sort::check(cudaMalloc(&d, n * 4));
  paged_sort::check(cudaMalloc(&original, n * 4));
  paged_sort::check(cudaMemcpy(original, x.data(), n * 4, cudaMemcpyHostToDevice));
  paged_sort::workspace scratch, merge_cache, radix_cache;
  paged_sort::statistics stats;
  paged_sort::policy p;
  p.scratch = &scratch;
  p.stats   = &stats;
  allocator ma{&merge_cache}, ra{&radix_cache};
  auto reset = [&] {
    paged_sort::check(cudaMemcpyAsync(d, original, n * 4, cudaMemcpyDeviceToDevice));
  };
  auto candidate = [&] {
    thrust::stable_sort(p, d, d + n);
  };
  auto merge = [&] {
    thrust::stable_sort(thrust::cuda::par(ma), d, d + n, numeric_less{});
  };
  auto radix = [&] {
    thrust::stable_sort(thrust::cuda::par(ra), d, d + n);
  };
  for (int i = 0; i < 5; ++i)
  {
    reset();
    merge();
    reset();
    candidate();
    reset();
    radix();
  }
  std::vector<float> m, c, r;
  for (int i = 0; i < repeats; ++i)
  {
    if (i % 2)
    {
      reset();
      c.push_back(time_it(candidate));
      reset();
      m.push_back(time_it(merge));
    }
    else
    {
      reset();
      m.push_back(time_it(merge));
      reset();
      c.push_back(time_it(candidate));
    }
    reset();
    r.push_back(time_it(radix));
  }
  reset();
  candidate();
  paged_sort::check(cudaMemcpy(x.data(), d, n * 4, cudaMemcpyDeviceToHost));
  reset();
  radix();
  std::vector<unsigned> want(n);
  paged_sort::check(cudaMemcpy(want.data(), d, n * 4, cudaMemcpyDeviceToHost));
  if (x != want)
  {
    throw std::runtime_error("scalar correctness mismatch");
  }
  std::sort(m.begin(), m.end());
  std::sort(c.begin(), c.end());
  std::sort(r.begin(), r.end());
  std::cout
    << "thrust_version,n,page_items,workers,merge_temp_bytes,radix_temp_bytes,candidate_"
       "temp_bytes,merge_ms,radix_ms,candidate_ms,vs_merge,vs_radix\n"
    << THRUST_VERSION << ',' << n << ',' << stats.page_items << ',' << stats.workers << ',' << merge_cache.capacity
    << ',' << radix_cache.capacity << ',' << scratch.capacity << ',' << m[repeats / 2] << ',' << r[repeats / 2] << ','
    << c[repeats / 2] << ',' << m[repeats / 2] / c[repeats / 2] << ',' << r[repeats / 2] / c[repeats / 2] << '\n';
  cudaFree(d);
  cudaFree(original);
}
catch (std::exception const& e)
{
  std::cerr << e.what() << '\n';
  return 1;
}
