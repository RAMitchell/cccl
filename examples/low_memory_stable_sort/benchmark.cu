// SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/version.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <random>
#include <vector>

#include "paged_stable_sort.cuh"

#ifndef RECORD_WORDS
#  define RECORD_WORDS 2
#endif
struct record
{
  uint32_t key, original;
#if RECORD_WORDS > 2
  uint32_t payload[RECORD_WORDS - 2];
#endif
  bool operator==(record const& x) const
  {
    bool same = key == x.key && original == x.original;
#if RECORD_WORDS > 2
    for (int j = 0; j < RECORD_WORDS - 2; ++j)
    {
      same &= payload[j] == x.payload[j];
    }
#endif
    return same;
  }
};
struct key_less
{
  bool descending = false;
  __host__ __device__ bool operator()(record a, record b) const
  {
    return descending ? a.key > b.key : a.key < b.key;
  }
};
struct allocation_counts
{
  size_t live = 0, peak = 0;
  paged_sort::workspace cache;
};
struct counting_allocator
{
  using value_type = char;
  allocation_counts* counts;
  char* allocate(std::ptrdiff_t n)
  {
    if (counts->live)
    {
      throw std::runtime_error("overlapping baseline allocations");
    }
    counts->cache.reserve(n);
    counts->live += n;
    counts->peak = std::max(counts->peak, counts->live);
    return static_cast<char*>(counts->cache.data);
  }
  void deallocate(char*, size_t n)
  {
    counts->live -= n;
  }
};
std::vector<record> input(size_t n, std::string dist, unsigned seed = 42)
{
  std::mt19937 rng(seed);
  std::vector<record> x(n);
  for (size_t i = 0; i < n; ++i)
  {
    uint32_t k = rng();
    if (dist == "duplicates")
    {
      k %= 17;
    }
    else if (dist == "equal")
    {
      k = 1;
    }
    else if (dist == "sorted")
    {
      k = i / 4;
    }
    else if (dist == "reverse")
    {
      k = (n - i) / 4;
    }
    else if (dist == "runs")
    {
      k = (i < n / 2 ? n / 2 + i : i - n / 2);
    }
    else if (dist == "alternating")
    {
      k = (i < n / 2 ? 2 * i : 2 * (i - n / 2) + 1);
    }
    x[i] = {k, uint32_t(i)};
#if RECORD_WORDS > 2
    for (int j = 0; j < RECORD_WORDS - 2; ++j)
    {
      x[i].payload[j] = uint32_t(i) * (j + 3) ^ 0x91e10da5;
    }
#endif
  }
  return x;
}
template <class F>
double timed(F f, cudaStream_t stream)
{
  cudaEvent_t a, b;
  paged_sort::check(cudaEventCreate(&a));
  paged_sort::check(cudaEventCreate(&b));
  paged_sort::check(cudaEventRecord(a, stream));
  f();
  paged_sort::check(cudaEventRecord(b, stream));
  paged_sort::check(cudaEventSynchronize(b));
  float ms;
  paged_sort::check(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms;
}
void test()
{
  cudaStream_t stream;
  paged_sort::check(cudaStreamCreate(&stream));
  int cases = 0;
  for (int b : {0, 1024, 2048, 4096, 8192, 16384, 32768, 65536})
  {
    for (bool desc : {false, true})
    {
      for (size_t n :
           {size_t(0),
            1UL,
            2UL,
            31UL,
            1023UL,
            1024UL,
            1025UL,
            2047UL,
            2048UL,
            2049UL,
            4095UL,
            4096UL,
            4097UL,
            8193UL,
            65537UL,
            100003UL})
      {
        for (auto dist : {"random", "duplicates", "equal", "sorted", "reverse", "runs", "alternating"})
        {
          auto x = input(n, dist), want = x;
          std::stable_sort(want.begin(), want.end(), key_less{desc});
          record* d;
          paged_sort::check(cudaMalloc(&d, std::max(size_t(1), n) * sizeof(record)));
          paged_sort::check(cudaMemcpyAsync(d, x.data(), n * sizeof(record), cudaMemcpyHostToDevice, stream));
          paged_sort::policy p;
          p.page_items = b;
          p.workers    = 8;
          p.stream     = stream;
          thrust::stable_sort(p, thrust::device_pointer_cast(d), thrust::device_pointer_cast(d) + n, key_less{desc});
          paged_sort::check(cudaMemcpy(x.data(), d, n * sizeof(record), cudaMemcpyDeviceToHost));
          cudaFree(d);
          if (x != want)
          {
            std::cerr << "FAILED n=" << n << " page=" << b << " dist=" << dist << " desc=" << desc << '\n';
            for (size_t i = 0; i < n; ++i)
            {
              if (!(x[i] == want[i]))
              {
                std::cerr << "at " << i << ": " << x[i].key << "," << x[i].original << " wanted " << want[i].key << ","
                          << want[i].original << '\n';
                break;
              }
            }
            std::exit(1);
          }
          ++cases;
        }
      }
    }
  }
  cudaStreamDestroy(stream);
  std::cout << "PASS " << cases << " exact stable-sort comparisons\n";
}
int main(int argc, char** argv)
try
{
  if (argc > 1 && std::string(argv[1]) == "--test")
  {
    test();
    return 0;
  }
  size_t n         = argc > 1 ? std::stoull(argv[1]) : 1 << 22;
  int b            = argc > 2 ? std::stoi(argv[2]) : 0;
  int workers      = argc > 3 ? std::stoi(argv[3]) : 0;
  std::string dist = argc > 4 ? argv[4] : "random";
  int repeats      = argc > 5 ? std::stoi(argv[5]) : 3;
  if (n == 0 || n > (size_t(1) << 31) || repeats < 1)
  {
    throw std::invalid_argument("benchmark requires 1 <= n <= 2^31 and positive repeats");
  }
  int warmups = std::getenv("LMS_WARMUPS") ? std::stoi(std::getenv("LMS_WARMUPS")) : 5;
  if (warmups < 0)
  {
    throw std::invalid_argument("negative warmups");
  }
  cudaDeviceProp prop{};
  paged_sort::check(cudaGetDeviceProperties(&prop, 0));
  std::cerr << prop.name << ", SMs=" << prop.multiProcessorCount << ", Thrust=" << THRUST_VERSION << "\n";
  auto x = input(n, dist);
  record *d, *source;
  paged_sort::check(cudaMalloc(&d, n * sizeof(record)));
  paged_sort::check(cudaMalloc(&source, n * sizeof(record)));
  paged_sort::check(cudaMemcpy(source, x.data(), n * sizeof(record), cudaMemcpyHostToDevice));
  paged_sort::statistics stats;
  paged_sort::policy p;
  p.page_items = b;
  p.workers    = workers;
  p.stats      = &stats;
  p.profile    = std::getenv("LMS_PROFILE") != nullptr;
  paged_sort::workspace scratch;
  p.scratch = &scratch;
  allocation_counts counts;
  counting_allocator alloc{&counts};
  auto reset = [&] {
    paged_sort::check(cudaMemcpyAsync(d, source, n * sizeof(record), cudaMemcpyDeviceToDevice));
  };
  auto baseline = [&] {
    thrust::stable_sort(thrust::cuda::par(alloc), d, d + n, key_less{});
  };
  auto candidate = [&] {
    thrust::stable_sort(p, d, d + n, key_less{});
  };
  for (int warm = 0; warm < warmups; ++warm)
  {
    reset();
    baseline();
    reset();
    candidate();
  }
  std::vector<double> base, cand;
  for (int r = 0; r < repeats; ++r)
  {
    if (r % 2 == 0)
    {
      reset();
      base.push_back(timed(baseline, 0));
      reset();
      cand.push_back(timed(candidate, 0));
    }
    else
    {
      reset();
      cand.push_back(timed(candidate, 0));
      reset();
      base.push_back(timed(baseline, 0));
    }
  }
  reset();
  candidate();
  paged_sort::check(cudaMemcpy(x.data(), d, n * sizeof(record), cudaMemcpyDeviceToHost));
  // Exact comparison against the same Thrust baseline also detects lost/duplicated payloads.
  reset();
  baseline();
  std::vector<record> want(n);
  paged_sort::check(cudaMemcpy(want.data(), d, n * sizeof(record), cudaMemcpyDeviceToHost));
  if (x != want)
  {
    throw std::runtime_error("benchmark correctness/stability mismatch");
  }
  std::sort(base.begin(), base.end());
  std::sort(cand.begin(), cand.end());
  std::cout << "n,distribution,record_bytes,page_items,workers,spare_pages,baseline_temp_bytes,"
               "candidate_temp_bytes,baseline_ms,candidate_ms,throughput_ratio\n";
  std::cout
    << n << ',' << dist << ',' << sizeof(record) << ',' << stats.page_items << ',' << stats.workers << ','
    << stats.spare_pages << ',' << counts.peak << ',' << stats.temporary_bytes << ',' << base[repeats / 2] << ','
    << cand[repeats / 2] << ',' << base[repeats / 2] / cand[repeats / 2] << '\n';
  if (p.profile)
  {
    std::cerr << "init=" << stats.init_ms << " block=" << stats.block_ms << " partition=" << stats.partition_ms
              << " merge=" << stats.merge_ms << " materialize=" << stats.materialize_ms << " ms\n";
  }
  cudaFree(source);
  cudaFree(d);
}
catch (std::exception const& e)
{
  std::cerr << e.what() << '\n';
  return 1;
}
