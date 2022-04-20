#include <array>
#include <vector>

#include <benchmark/benchmark.h>

static void find_string_inv_bench(benchmark::State& state)
{
  // Code inside this loop is measured repeatedly
  auto range = state.range(0);

  const auto array_size = 1001;
  std::array<char, array_size> tmp = {0};

  for (auto _ : state) {
    tmp[range] = 1;
    benchmark::DoNotOptimize(tmp);
    // benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations());
  // state.SetBytesProcessed(state.iterations() * state.range(0) *
  // sizeof(char));
}
// Register the function as a benchmark
BENCHMARK(find_string_inv_bench)
    ->Name("find_string_inv")
    ->RangeMultiplier(10)
    ->Range(1, 1000);

// Run the benchmark
// BENCHMARK_MAIN();
