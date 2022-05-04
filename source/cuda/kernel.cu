//////////////////////////////////////////////////////////////
//   ____                                                   //
//  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___  //
//  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __| //
//  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__  //
//  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___| //
//                             |_|             |_|          //
//////////////////////////////////////////////////////////////
//                                                          //
//  BenLib, 2021                                            //
//  Created: 21, March, 2021                                //
//  Modified: 29, April, 2022                               //
//  file: kernel.cu                                         //
//  Crypto                                                  //
//  Source:
//  https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference
//  //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/ //
//          https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#asynchronous-transfers-and-overlapping-transfers-with-computation__concurrent-copy-and-execute
//          https://www.ce.jhu.edu/dalrymple/classes/602/Class12.pdf //
//          https://create.stephan-brumme.com/crc32/
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda_runtime.h>

#include "kernel.cuh"
#include "kernel.hpp"

__host__ void dummy_gpu()
{
  dummy_gpu_kernel<<<128, 128>>>();
}

__global__ void dummy_gpu_kernel()
{
  // do something
}

__host__ void jamcrc_wrapperv2(dim3& grid,
                               dim3& threads,
                               cudaStream_t& stream,
                               const int& device,
                               const void* data,
                               uint32_t& result,
                               uint64_t& length,
                               const uint32_t& previousCrc32)
{
  jamcrc_kernel_wrapperv2<<<grid, threads, device, stream>>>(data, result, length, previousCrc32);
}

__global__ void jamcrc_kernel_wrapperv2(const void* data,
                                        uint32_t& result,
                                        uint64_t& length,
                                        const uint32_t& previousCrc32)
{
  const uint64_t id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id <= 0) {
    result = jamcrc_kernelv2(data, length, previousCrc32);
  }
}

__device__ uint32_t jamcrc_kernelv2(const void* data, uint64_t& length, const uint32_t& previousCrc32)
{
  uint32_t crc = ~previousCrc32;
  unsigned char* current = (unsigned char*)data;
  while (length--)
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  return crc;
}

__host__ uint32_t my::cuda::jamcrcv2(const void* data, uint64_t& length, uint32_t& previousCrc32)
{
  uint cuda_block_size = 32;

  int device = 0;
  cudaGetDevice(&device);
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Calculate length of the array with max_range and min_range
  uint64_t data_size = length * sizeof(void*);

  uint32_t* data_cuda = nullptr;

  cudaMallocManaged(&data_cuda, data_size, cudaMemAttachGlobal);

  cudaStreamAttachMemAsync(stream, &data_cuda);

  cudaMemPrefetchAsync(data_cuda, data_size, device, stream);

  // std::copy(data, data + length, data_cuda);
  memcpy(data_cuda, data, data_size);

  uint64_t grid_size = static_cast<uint64_t>(ceil(static_cast<double>(data_size) / cuda_block_size));

  dim3 threads(static_cast<uint>(cuda_block_size), 1, 1);
  dim3 grid(static_cast<uint>(grid_size), 1, 1);

  uint32_t result = 0;

  jamcrc_wrapperv2(grid, threads, stream, device, data_cuda, result, length, previousCrc32);

  cudaStreamSynchronize(stream);
  cudaDeviceSynchronize();

  cudaFree(data_cuda);
  cudaStreamDestroy(stream);

  return result;
}

__host__ void my::cuda::launch_kernelv2(std::vector<uint32_t>& jamcrc_results,
                                        std::vector<uint64_t>& index_results,
                                        const uint64_t& min_range,
                                        const uint64_t& max_range,
                                        const uint64_t& cuda_block_size)
{
  // int device = -1;
  // cudaGetDevice(&device);

  int device = 0;
  cudaGetDevice(&device);

  /*
  int priority_high, priority_low;
  cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
  cudaStream_t st_high, st_low;
  cudaStreamCreateWithPriority(&st_high, cudaStreamNonBlocking, priority_high);
  cudaStreamCreateWithPriority(&st_low, cudaStreamNonBlocking, priority_low);
  */

  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

  // Calculate length of the array with max_range and min_range
  uint64_t array_length = static_cast<uint64_t>((max_range - min_range) / 20000000 + 1);
  uint64_t jamcrc_results_size = array_length * sizeof(uint32_t);
  uint64_t index_results_size = array_length * sizeof(uint64_t);

  uint32_t* jamcrc_results_ptr = nullptr;
  uint64_t* index_results_ptr = nullptr;

  cudaMallocManaged(&jamcrc_results_ptr, jamcrc_results_size, cudaMemAttachGlobal);
  cudaMallocManaged(&index_results_ptr, index_results_size, cudaMemAttachGlobal);

  cudaStreamAttachMemAsync(stream, &jamcrc_results_ptr);
  cudaStreamAttachMemAsync(stream, &index_results_size);

  cudaMemPrefetchAsync(jamcrc_results_ptr, jamcrc_results_size, device, stream);
  cudaMemPrefetchAsync(index_results_ptr, index_results_size, device, stream);

  for (uint64_t i = 0; i < array_length; ++i) {
    jamcrc_results_ptr[i] = 0;
    index_results_ptr[i] = 0;
  }

  uint64_t grid_size = static_cast<uint64_t>(ceil(static_cast<double>(max_range - min_range) / cuda_block_size));
  // std::cout << "CUDA Grid size: " << grid_size << std::endl;
  // std::cout << "CUDA Block size: " << cuda_block_size << std::endl;

  dim3 threads(static_cast<uint>(cuda_block_size), 1, 1);
  dim3 grid(static_cast<uint>(grid_size), 1, 1);

  my::cuda::launch_kernelv2(
      grid, threads, stream, jamcrc_results_ptr, index_results_ptr, array_length, min_range, max_range);

  // my::cuda::launch_kernel();
  cudaStreamSynchronize(stream);

  for (uint64_t i = 0; i < array_length; ++i) {
    if (jamcrc_results_ptr[i] != index_results_ptr[i]) {
      jamcrc_results.emplace_back(jamcrc_results_ptr[i]);
      index_results.emplace_back(index_results_ptr[i]);
    }
  }

  cudaDeviceSynchronize();
  cudaFree(jamcrc_results_ptr);
  cudaFree(index_results_ptr);

  cudaStreamDestroy(stream);
  // cudaStreamDestroy(st_high);
  // cudaStreamDestroy(st_low);
}

__host__ void my::cuda::launch_kernelv2(size_t grid,
                                        size_t threads,
                                        cudaStream_t& stream,
                                        uint32_t* crc_result,
                                        uint64_t* index_result,
                                        uint64_t array_size,
                                        uint64_t a,
                                        uint64_t b)
{
  runner_kernelv2<<<grid, threads, 0, stream>>>(crc_result, index_result, array_size, a, b);
}

__host__ void my::cuda::launch_kernelv2(dim3& grid,
                                        dim3& threads,
                                        cudaStream_t& stream,
                                        uint32_t* crc_result,
                                        uint64_t* index_result,
                                        uint64_t array_size,
                                        uint64_t a,
                                        uint64_t b)
{
  runner_kernelv2<<<grid, threads, 0, stream>>>(crc_result, index_result, array_size, a, b);
}

__global__ void runner_kernelv2(
    uint32_t* crc_result, uint64_t* index_result, uint64_t array_size, uint64_t a, uint64_t b)
{
  const uint64_t id = blockIdx.x * blockDim.x + threadIdx.x + a;
  // printf("blockIdx %d, blockDimx %d, threadIdx %d\n", blockIdx.x, blockDim.x, threadIdx.x);

  if (id <= b && id >= a) {
    // printf("blockIdx %d, blockDim %d, threadIdx %d\n", blockIdx.x, blockDim.x, threadIdx.x);

    // Allocate memory for the array
    unsigned char array[29] = {0};

    uint64_t size = 0;
    // Generate the array
    find_string_inv_kernelv2(array, id, size);

    // Calculate the CRC
    uint32_t previousCrc32 = 0;
    uint32_t result = jamcrc_kernelv2(array, size, previousCrc32);

    bool found = false;
    for (uint32_t i = 0; i < 87; i++) {
      if (result == cheat_list[i]) {
        found = true;
        break;
      }
    }

    if (!found) {
      return;
    }

    //__syncthreads();

    for (uint64_t i = 0; i < array_size; i++) {
      if (crc_result[i] == 0 && index_result[i] == 0) {
        crc_result[i] = result;
        index_result[i] = id;
        break;
      }
    }
  }
}

__device__ void find_string_inv_kernelv2(unsigned char* array, uint64_t n, uint64_t& terminator_index)
{
  const uint32_t string_size_alphabet = 27;

  const unsigned char alpha[string_size_alphabet] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  // If n < 27
  if (n < 26) {
    array[0] = alpha[n];
    array[1] = '\0';
    terminator_index = 1;
    return;
  }
  // If n > 27
  uint64_t i = 0;
  while (n > 0) {
    array[i] = alpha[(--n) % 26];
    n /= 26;
    ++i;
  }
  array[i] = '\0';
  terminator_index = i;
}
