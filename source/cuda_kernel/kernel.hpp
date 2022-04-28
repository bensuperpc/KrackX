
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
//  Modified: 17, March, 2021                               //
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

#ifndef _CUDA_KERNEL_HPP_
#define _CUDA_KERNEL_HPP_

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

#include "stdio.h"

namespace my
{
namespace cuda
{
void JAMCRC_byte_tableless2(const dim3& grid,
                            const dim3& threads,
                            cudaStream_t& stream,
                            unsigned char* data,
                            ulong length,
                            uint previousCrc32,
                            uint* resultCrc32);
void JAMCRC_byte_tableless2(
    const dim3& grid, const dim3& threads, unsigned char* data, ulong length, uint previousCrc32, uint* resultCrc32);
void vecMult(size_t gridSize, size_t blockSize, int* a, int* b, int* c, size_t n);
void vecMult(size_t gridSize, size_t blockSize, cudaStream_t& stream, int* a, int* b, int* c, size_t n);
void matrixMultiplySimple(dim3 gridSize, dim3 blockSize, int* a, int* b, int* c, size_t n);
void matrixMultiplySimple(dim3 gridSize, dim3 blockSize, cudaStream_t& stream, int* a, int* b, int* c, size_t n);

}  // namespace cuda
}  // namespace my

#endif