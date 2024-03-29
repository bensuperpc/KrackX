#include <array>
#include <memory>
#include <string>
#include <vector>

#include <benchmark/benchmark.h>

#include "gta_sa.hpp"

#ifdef BUILD_WITH_CUDA
#  include "cuda/wrapper.hpp"
#endif

// Source: https://github.com/stbrumme/crc32
static std::unique_ptr<char[]> generate(const std::uint64_t length);

static std::unique_ptr<char[]> generate(const std::uint64_t length)
{
  uint32_t randomNumber = 0x27121978;
  // initialize
  std::unique_ptr<char[]> data(new char[length]);
  for (size_t i = 0; i < length; i++) {
    data[i] = char(randomNumber & 0xFF);
    // simple LCG, see
    // http://en.wikipedia.org/wiki/Linear_congruential_generator
    randomNumber = 1664525 * randomNumber + 1013904223;
  }
  return data;
}

static void jamcrc_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  auto size = state.range(0);
  auto c_str = generate(size);

  std::string str;
  str.reserve();
  for (auto i = 0; i < size; i++) {
    str.push_back(c_str[i] + '0');
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(str);
    GTA_SA::jamcrc(str);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(char));

  auto c = c_str.release();
  delete[] c;
}
BENCHMARK(jamcrc_bench)->Name("jamcrc")->RangeMultiplier(2)->Range(1, 32);

#ifdef BUILD_WITH_CUDA
static void jamcrc_cuda_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  auto size = state.range(0);
  std::cout << "size: " << size << std::endl;
  auto c_str = generate(size);

  std::string str;
  str.reserve();
  for (auto i = 0; i < size; i++) {
    str.push_back(c_str[i] + '0');
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(str);
    my::cuda::jamcrc(str.data(), str.size(), 0);
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  state.SetBytesProcessed(state.iterations() * state.range(0) * sizeof(char));

  auto c = c_str.release();
  delete[] c;
}
BENCHMARK(jamcrc_cuda_bench)->Name("jamcrc_cuda")->RangeMultiplier(2)->Range(1, 32);
#endif

// Run the benchmark
// BENCHMARK_MAIN();
