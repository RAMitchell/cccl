// SPDX-FileCopyrightText: Copyright (c) 2026 Rory Mitchell
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/version.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "paged_radix_sort.cuh"
#ifndef LMR_RADIX_BITS
#  define LMR_RADIX_BITS 8
#endif
#ifndef LMR_PAGE_ITEMS
#  define LMR_PAGE_ITEMS 4096
#endif
using paged_sort::check;
using radix_policy = paged_radix::policy<LMR_RADIX_BITS, LMR_PAGE_ITEMS>;
struct allocator
{
  using value_type = char;
  paged_sort::workspace* scratch;
  char* allocate(std::ptrdiff_t n)
  {
    scratch->reserve(n);
    return static_cast<char*>(scratch->data);
  }
  void deallocate(char*, size_t) {}
};
struct item
{
  unsigned key, sequence;
  bool operator==(item const& other) const
  {
    return key == other.key && sequence == other.sequence;
  }
};
struct item_key
{
  __host__ __device__ unsigned operator()(item x) const
  {
    return x.key;
  }
};
struct key_less
{
  bool operator()(item a, item b) const
  {
    return a.key < b.key;
  }
};
unsigned make_key(size_t i, size_t n, std::string const& distribution, std::mt19937& rng)
{
  if (distribution == "random")
  {
    return rng();
  }
  if (distribution == "duplicates")
  {
    return rng() % 17;
  }
  if (distribution == "equal")
  {
    return 42;
  }
  if (distribution == "sorted")
  {
    return unsigned(i);
  }
  if (distribution == "reverse")
  {
    return unsigned(n - i);
  }
  if (distribution == "alternating")
  {
    return i % 2 ? UINT_MAX : 0;
  }
  throw std::invalid_argument("unknown distribution");
}
void check_case(int n, int workers, std::string const& distribution, cudaStream_t stream, paged_sort::workspace& scratch)
{
  std::mt19937 rng(731);
  paged_radix::policy<LMR_RADIX_BITS, LMR_PAGE_ITEMS, item_key> p;
  p.scratch = &scratch;
  p.workers = workers;
  std::vector<item> x(n + 6, {0xdeadbeef, 0xbaadf00d});
  for (int i = 0; i < n; ++i)
  {
    x[i + 3] = {make_key(i, n, distribution, rng), unsigned(i)};
  }
  auto want = x;
  std::stable_sort(want.begin() + 3, want.end() - 3, key_less{});
  item* d;
  check(cudaMalloc(&d, x.size() * sizeof(item)));
  check(cudaMemcpyAsync(d, x.data(), x.size() * sizeof(item), cudaMemcpyHostToDevice, stream));
  // Exercise raw pointers, device_ptr, and temporary policies on a nondefault stream.
  if (workers % 2)
  {
    thrust::stable_sort(p.on(stream), thrust::device_pointer_cast(d + 3), thrust::device_pointer_cast(d + 3 + n));
  }
  else
  {
    thrust::stable_sort(p.on(stream), d + 3, d + 3 + n);
  }
  check(cudaMemcpyAsync(x.data(), d, x.size() * sizeof(item), cudaMemcpyDeviceToHost, stream));
  check(cudaStreamSynchronize(stream));
  check(cudaFree(d));
  if (x != want)
  {
    std::cerr << "failed " << n << ' ' << distribution << " workers=" << workers << '\n';
    for (int i = 0; i < n + 6; ++i)
    {
      if (!(x[i] == want[i]))
      {
        std::cerr << "at " << i << ": " << x[i].key << ',' << x[i].sequence << " expected " << want[i].key << ','
                  << want[i].sequence << '\n';
        break;
      }
    }
    throw std::runtime_error("stable reference mismatch");
  }
}
void correctness(bool stress)
{
  int cases = 0;
  cudaStream_t stream;
  check(cudaStreamCreate(&stream));
  paged_sort::workspace scratch;
  std::vector<int> sizes =
    stress ? std::vector<int>{4194304, 16777219}
           : std::vector<int>{0, 1, 2, 31, 255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 65539, 262147};
  for (int workers : {1, 3, 17, 376, 752})
  {
    for (int n : sizes)
    {
      for (auto distribution : {"random", "duplicates", "equal", "sorted", "reverse", "alternating"})
      {
        check_case(n, workers, distribution, stream, scratch);
        ++cases;
      }
    }
  }
  // Validate the runtime worker constraint even on an empty range.
  radix_policy p;
  unsigned empty = 0;
  for (int workers : {-1, 65537})
  {
    p.workers     = workers;
    bool rejected = false;
    try
    {
      thrust::stable_sort(p, &empty, &empty);
    }
    catch (std::invalid_argument const&)
    {
      rejected = true;
    }
    if (!rejected)
    {
      throw std::runtime_error("invalid worker count accepted");
    }
  }
  check(cudaStreamDestroy(stream));
  std::cout << "PASS " << cases << " exact stable reference and guarded-range cases; invalid workers rejected\n";
}
template <class F>
float time_it(F f, cudaEvent_t begin, cudaEvent_t end)
{
  check(cudaEventRecord(begin));
  f();
  check(cudaEventRecord(end));
  check(cudaEventSynchronize(end));
  float ms;
  check(cudaEventElapsedTime(&ms, begin, end));
  return ms;
}
int main(int argc, char** argv)
try
{
  if (argc > 1 && std::string(argv[1]) == "--test")
  {
    correctness(false);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "--stress")
  {
    correctness(true);
    return 0;
  }
  if (argc > 1 && std::string(argv[1]) == "--check")
  {
    if (argc != 5)
    {
      throw std::invalid_argument("--check requires n, workers, distribution");
    }
    int n = std::stoi(argv[2]), workers = std::stoi(argv[3]);
    if (n < 0 || n > INT_MAX - 6)
    {
      throw std::invalid_argument("invalid check length");
    }
    cudaStream_t stream;
    check(cudaStreamCreate(&stream));
    paged_sort::workspace scratch;
    check_case(n, workers, argv[4], stream, scratch);
    check(cudaStreamDestroy(stream));
    std::cout << "PASS stable reference, guards, and nondefault stream\n";
    return 0;
  }
  size_t n                 = argc > 1 ? std::stoull(argv[1]) : 1 << 24;
  int repeats              = argc > 2 ? std::stoi(argv[2]) : 5;
  int workers              = argc > 3 ? std::stoi(argv[3]) : 0;
  std::string distribution = argc > 4 ? argv[4] : "random";
  if (n < 2 || n > INT_MAX || repeats < 1)
  {
    throw std::invalid_argument("invalid n or repeats");
  }
  std::vector<unsigned> host(n);
  std::mt19937 rng(42);
  for (size_t i = 0; i < n; ++i)
  {
    host[i] = make_key(i, n, distribution, rng);
  }
  unsigned *data, *original;
  check(cudaMalloc(&data, n * 4));
  check(cudaMalloc(&original, n * 4));
  check(cudaMemcpy(original, host.data(), n * 4, cudaMemcpyHostToDevice));
  paged_sort::workspace candidate_scratch, baseline_scratch;
  paged_radix::statistics stats;
  radix_policy p;
  p.scratch  = &candidate_scratch;
  p.stats    = &stats;
  p.workers  = workers;
  auto reset = [&] {
    check(cudaMemcpyAsync(data, original, n * 4, cudaMemcpyDeviceToDevice));
  };
  auto candidate = [&] {
    thrust::stable_sort(p, data, data + n);
  };
  auto baseline = [&] {
    thrust::stable_sort(thrust::cuda::par(allocator{&baseline_scratch}), data, data + n);
  };
  int warmups = std::getenv("LMS_WARMUPS") ? std::stoi(std::getenv("LMS_WARMUPS")) : 5;
  for (int i = 0; i < warmups; ++i)
  {
    reset();
    baseline();
    reset();
    candidate();
  }
  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));
  std::vector<float> a, b;
  for (int i = 0; i < repeats; ++i)
  {
    if (i % 2)
    {
      reset();
      b.push_back(time_it(candidate, begin, end));
      reset();
      a.push_back(time_it(baseline, begin, end));
    }
    else
    {
      reset();
      a.push_back(time_it(baseline, begin, end));
      reset();
      b.push_back(time_it(candidate, begin, end));
    }
  }
  reset();
  candidate();
  check(cudaMemcpy(host.data(), data, n * 4, cudaMemcpyDeviceToHost));
  reset();
  baseline();
  std::vector<unsigned> want(n);
  check(cudaMemcpy(want.data(), data, n * 4, cudaMemcpyDeviceToHost));
  if (host != want)
  {
    throw std::runtime_error("Thrust comparison mismatch");
  }
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::cout
    << "thrust_version,n,distribution,digit_bits,page_items,block_threads,workers,spare_pages,baseline_temp_bytes,"
       "candidate_temp_bytes,baseline_peak_bytes,candidate_peak_bytes,baseline_ms,candidate_ms,throughput_ratio\n"
    << THRUST_VERSION << ',' << n << ',' << distribution << ',' << stats.digit_bits << ',' << stats.page_items << ','
    << stats.block_threads << ',' << stats.workers << ',' << stats.spare_pages << ',' << baseline_scratch.capacity
    << ',' << candidate_scratch.capacity << ',' << n * 4 + baseline_scratch.capacity << ','
    << n * 4 + candidate_scratch.capacity << ',' << a[repeats / 2] << ',' << b[repeats / 2] << ','
    << a[repeats / 2] / b[repeats / 2] << '\n';
  if (std::getenv("LMS_PROFILE"))
  {
    p.profile = true;
    reset();
    candidate();
    std::cerr
      << "init=" << stats.init_ms << " histogram=" << stats.histogram_ms << " pass_setup=" << stats.pass_setup_ms
      << " scatter=" << stats.scatter_ms << " materialize=" << stats.materialize_ms << '\n';
  }
  check(cudaEventDestroy(begin));
  check(cudaEventDestroy(end));
  check(cudaFree(data));
  check(cudaFree(original));
}
catch (std::exception const& e)
{
  std::cerr << e.what() << '\n';
  return 1;
}
