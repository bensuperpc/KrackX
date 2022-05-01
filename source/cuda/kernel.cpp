
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
//  Created: 16, March, 2021                                //
//  Modified: 29, April, 2022                               //
//  file: kernel.h                                          //
//  Crypto                                                  //
//  Source:
//  https://stackoverflow.com/questions/13553015/cuda-c-linker-error-undefined-reference
//  //
//          https://www.olcf.ornl.gov/tutorials/cuda-vector-addition/ //
//          https://gist.github.com/AndiH/2e2f6cd9bccd64ec73c3b1d2d18284e0
//          https://stackoverflow.com/a/14038590/10152334
//          https://www.daniweb.com/programming/software-development/threads/292133/convert-1d-array-to-2d-array
//          https://www.daniweb.com/programming/software-development/threads/471477/equivalent-iteration-of-2d-and-3d-array-flattened-as-1d-array
//          http://coliru.stacked-crooked.com/a/7c570672c13ca3bf
//          https://math.stackexchange.com/questions/63074/is-there-a-3-dimensional-matrix-by-matrix-product
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#include <cuda/kernel.hpp>

__host__ void my::cuda::launch_kernel(std::vector<uint32_t>& _jamcrc_results,
                                      std::vector<uint64_t>& _index_results,
                                      const uint64_t& min_range,
                                      const uint64_t& max_range,
                                      const uint64_t& cuda_block_size)
{
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
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128 * 1024 * 1024);

  // Calculate length of the array with max_range and min_range
  auto array_length = (max_range - min_range) / 20000000 + 1;
  auto jamcrc_results_size = array_length * sizeof(uint32_t);
  auto index_results_size = array_length * sizeof(uint64_t);

  uint32_t* jamcrc_results = 0;
  uint64_t* index_results = 0;

  cudaMallocManaged(&jamcrc_results, jamcrc_results_size, cudaMemAttachGlobal);
  cudaMallocManaged(&index_results, index_results_size, cudaMemAttachGlobal);

  cudaStreamAttachMemAsync(stream, &jamcrc_results);
  cudaStreamAttachMemAsync(stream, &index_results_size);

  cudaMemPrefetchAsync(jamcrc_results, jamcrc_results_size, cuda_device, stream);
  cudaMemPrefetchAsync(index_results, index_results_size, cuda_device, stream);

  for (int i = 0; i < array_length; ++i) {
    jamcrc_results[i] = 0;
    index_results[i] = 0;
  }

  size_t grid_size = (int)ceil((float)(max_range - min_range) / cuda_block_size);
  // std::cout << "CUDA Grid size: " << grid_size << std::endl;
  // std::cout << "CUDA Block size: " << cuda_block_size << std::endl;

  dim3 threads(cuda_block_size, 1, 1);
  dim3 grid(grid_size, 1, 1);

  cudaStreamSynchronize(stream);
  my::cuda::launch_kernel(grid, threads, stream, jamcrc_results, index_results, array_length, min_range, max_range);

  // my::cuda::launch_kernel();
  cudaStreamSynchronize(stream);

  for (int i = 0; i < array_length; ++i) {
    if (jamcrc_results[i] != index_results[i]) {
      _jamcrc_results.emplace_back(jamcrc_results[i]);
      _index_results.emplace_back(index_results[i]);
    }
  }

  cudaDeviceSynchronize();
  cudaFree(jamcrc_results);
  cudaFree(index_results);

  cudaStreamDestroy(stream);
  cudaStreamDestroy(st_high);
  cudaStreamDestroy(st_low);
}
