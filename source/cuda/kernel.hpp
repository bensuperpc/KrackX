
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

#ifndef CUDA_KERNEL_HPP
#define CUDA_KERNEL_HPP

#include <cmath>
#include <cstdio>
#include <cstring>  // memcpy
#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
namespace my
{
namespace cuda
{
__host__ void launch_kernel(size_t grid,
                            size_t threads,
                            cudaStream_t& stream,
                            uint32_t* crc_result,
                            uint64_t* index_result,
                            uint64_t array_size,
                            uint64_t a,
                            uint64_t b);

__host__ void launch_kernel(dim3& grid,
                            dim3& threads,
                            cudaStream_t& stream,
                            uint32_t* crc_result,
                            uint64_t* index_result,
                            uint64_t array_size,
                            uint64_t a,
                            uint64_t b);

__host__ void launch_kernel(std::vector<uint32_t>& jamcrc_results,
                            std::vector<uint64_t>& index_results,
                            const uint64_t& min_range,
                            const uint64_t& max_range,
                            const uint64_t& cuda_block_size);

__host__ void jamcrc_wrapper(dim3& grid,
                             dim3& threads,
                             cudaStream_t& stream,
                             const int& device,
                             const void* data,
                             uint32_t& result,
                             uint64_t& length,
                             const uint32_t& previousCrc32);

__host__ uint32_t jamcrc(const void* data, uint64_t& length, uint32_t& previousCrc32);

}  // namespace cuda
}  // namespace my

#endif