#include "gta_sa.h"

GTA_SA::GTA_SA() {}

void GTA_SA::clear()
{
  this->results.clear();
  this->results.shrink_to_fit();
}

void GTA_SA::run()
{
#if !defined(_OPENMP)
  if (this->use_openmp) {
    std::cout << "OpenMP is not enabled, please compile with -fopenmp flag or "
                 "change use_openmp to false, fall back to std::thread."
                 "Doesn't impact performance most of the time."
              << std::endl;
    use_openmp = false;
  }
#endif

  if (this->use_openmp) {
    std::cout << "Running with OpenMP" << std::endl;
  } else {
    std::cout << "Running with std::thread" << std::endl;
  }

  std::cout << "Max thread support: " << this->max_thread_support() << std::endl;
  std::cout << "Running with: " << this->num_thread << " threads" << std::endl;

  std::array<char, 29> tmp1 = {0};
  std::array<char, 29> tmp2 = {0};

  if (this->min_range > this->max_range) {
    std::cout << "Min range value: '" << this->min_range << "' can't be greater than Max range value: '"
              << this->max_range << "'" << std::endl;
    return;
  }

  std::cout << "Number of calculations: " << (this->max_range - this->min_range) << std::endl;
  std::cout << "" << std::endl;

  GTA_SA::find_string_inv(tmp1.data(), this->min_range);
  GTA_SA::find_string_inv(tmp2.data(), this->max_range);
  std::cout << "From: " << tmp1.data() << " to: " << tmp2.data() << " Alphabetic sequence" << std::endl;
  std::cout << "" << std::endl;

  if (this->use_openmp) {
#if defined(_OPENMP)
    omp_set_num_threads(num_thread);
#endif

    this->begin_time = std::chrono::high_resolution_clock::now();
#if defined(_OPENMP)
#  ifdef _MSC_VER
    std::int64_t i = 0;  // OpenMP (2.0) on Windows doesn't support unsigned variable
#    pragma omp parallel for shared(results) schedule(dynamic)
// firstprivate(tmp, crc)
#  else
    std::uint64_t i = 0;
#    pragma omp parallel for schedule(auto) shared(results)
// firstprivate(tmp, crc)
#  endif
#else
    std::int64_t i = 0;
#endif

    for (i = min_range; i <= max_range; i++) {
      runner(i);
    }
  } else {
    thread_pool pool(num_thread);
    pool.parallelize_loop(min_range,
                          max_range,
                          [&](const std::uint64_t& min_range, const std::uint64_t& max_range)
                          {
                            for (std::uint64_t i = min_range; i <= max_range; i++) {
                              runner(i);
                            }
                          });
  }
  this->end_time = std::chrono::high_resolution_clock::now();

#if defined(BUILD_WITH_CUDA)
  // int device = -1;
  // cudaGetDevice(&device);

  const uint32_t cuda_device = 0;

  if (cudaSetDevice(cuda_device) != cudaSuccess) {
    std::cout << "Error on cudaSetDevice" << std::endl;
  }

  int priority_high, priority_low;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
  cudaStream_t st_high, st_low;
  cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
  cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
  cudaStream_t stream;
  // cudaStreamAttachMemAsync(stream1, &x);
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  auto block_size = 512;
  auto array_length = 32;
  auto array_size = array_length * sizeof(int);

  int *MatA, *MatB, *MatC;

  cudaMallocManaged(&MatA, array_size, cudaMemAttachGlobal);
  cudaMallocManaged(&MatB, array_size, cudaMemAttachGlobal);
  cudaMallocManaged(&MatC, array_size, cudaMemAttachGlobal);

  // cudaStreamSynchronize(stream);

  cudaMemPrefetchAsync(MatA, array_size, cuda_device, stream);
  cudaMemPrefetchAsync(MatB, array_size, cuda_device, stream);
  cudaMemPrefetchAsync(MatC, array_size, cuda_device, stream);

  for (int i = 0; i < array_length; ++i) {
    MatA[i] = 8;
    MatB[i] = 4;
    MatC[i] = 0;
  }

  // dim3 threads(block_size, block_size);
  // dim3 grid(array_length / threads.x, array_length / threads.y);
  // my::cuda::matrixMultiplySimple(grid, threads, stream, MatA, MatB, MatC, 32);

  size_t grid = (int)ceil((float)array_length / block_size);
  size_t threads = block_size;
  my::cuda::vecMult(grid, threads, stream, MatA, MatB, MatC, 32);

  cudaStreamSynchronize(stream);

  for (int i = 0; i < array_length; ++i) {
    std::cout << MatA[i] << std::endl;
  }

  for (int i = 0; i < array_length; ++i) {
    std::cout << MatB[i] << std::endl;
  }

  for (int i = 0; i < array_length; ++i) {
    std::cout << MatC[i] << std::endl;
  }

  cudaFree(MatA);
  cudaFree(MatB);
  cudaFree(MatC);

  cudaStreamDestroy(stream);
  cudaStreamDestroy(st_high);
  cudaStreamDestroy(st_low);

#endif

  std::sort(this->results.begin(), this->results.end());  // Sort results

  constexpr auto display_val = 18;

  std::cout << std::setw(display_val + 3) << "Iter. N°" << std::setw(display_val) << "Code"
            << std::setw(display_val + 8) << "JAMCRC value" << std::endl;

  for (auto& result : this->results) {
    std::cout << std::setw(display_val + 3) << std::get<0>(result) << std::setw(display_val) << std::get<1>(result)
              << std::setw(display_val) << "0x" << std::hex << std::get<2>(result) << std::dec << std::endl;
  }
  std::cout << "Time: "
            << std::chrono::duration_cast<std::chrono::duration<double>>(this->end_time - this->begin_time).count()
            << " sec" << std::endl;  // Display time

  std::cout << "This program execute: " << std::fixed
            << (static_cast<double>(this->max_range - this->min_range)
                / std::chrono::duration_cast<std::chrono::duration<double>>(this->end_time - this->begin_time).count())
          / 1000000
            << " MOps/sec" << std::endl;  // Display perf
}

void GTA_SA::runner(const std::uint64_t& i)
{
  std::array<char, 29> tmp = {0};
  GTA_SA::find_string_inv(tmp.data(),
                          i);  // Generate Alphabetic sequence from uint64_t
                               // value, A=1, Z=27, AA = 28, AB = 29
  uint32_t crc = GTA_SA::jamcrc(tmp.data());  // JAMCRC

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 202002L) \
     || __cplusplus >= 202002L && !defined(ANDROID) && !defined(__EMSCRIPTEN__) && !defined(__clang__))

  const auto&& it = std::find(std::execution::unseq, std::begin(GTA_SA::cheat_list), std::end(GTA_SA::cheat_list), crc);

#else
  const auto&& it = std::find(std::begin(GTA_SA::cheat_list), std::end(GTA_SA::cheat_list), crc);
#endif

  // If crc is present in Array
  if (it != std::end(GTA_SA::cheat_list)) {
    std::reverse(tmp.data(),
                 tmp.data() + strlen(tmp.data()));  // Invert char array

    const auto index = it - std::begin(GTA_SA::cheat_list);

    this->results.emplace_back(std::make_tuple(i,
                                               std::string(tmp.data()),
                                               crc,
                                               cheat_list_name.at(index)));  // Save result: calculation position,
                                                                             // Alphabetic sequence, CRC,
  }
}

#if ((defined(_MSVC_LANG) && _MSVC_LANG >= 201703L) || __cplusplus >= 201703L)
auto GTA_SA::jamcrc(std::string_view my_string) -> std::uint32_t
{
#else

#  if _MSC_VER && !__INTEL_COMPILER
#    pragma message("C++17 is not enabled, the program will be less efficient with previous standards")
#  else
#    warning C++17 is not enabled, the program will be less efficient with previous standards.
#  endif

auto GTA_SA::jamcrc(const std::string& my_string) -> std::uint32_t
{
#endif
  auto crc = static_cast<uint32_t>(-1);
  const auto* current = reinterpret_cast<const unsigned char*>(my_string.data());
  uint64_t length = my_string.length();
  // process eight bytes at once
  while (static_cast<bool>(length--)) {
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  }
  return crc;
}

/**
 * \brief Generate Alphabetic sequence from uint64_t value, A=0, Z=26, AA = 27,
 * T \param n index in base 26 \param array return array
 */
void GTA_SA::find_string_inv(char* array, uint64_t n)
{
  constexpr std::uint32_t string_size_alphabet {27};
  constexpr std::array<char, string_size_alphabet> alpha {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  // If n < 27
  if (n < 26) {
    array[0] = alpha[n];
    return;
  }
  // If n > 27
  std::uint64_t i = 0;
  while (n > 0) {
    array[i] = alpha[(--n) % 26];
    n /= 26;
    ++i;
  }
}
