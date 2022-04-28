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
//  Modified: 25, March, 2021                               //
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

extern "C" {
// Can be remove if use "external *Function*" in .c/cpp file
#include "kernel.h"
}
#include "kernel.cuhpp"
#include "kernel.hpp"

__global__ void JAMCRC_byte_tableless2_kernel(uchar* data, ulong length, uint previousCrc32, uint* resultCrc32)
{
  uint crc = ~previousCrc32;

  while (length-- != 0) {
    crc = crc ^ *data++;
    uint c = (((crc << 31) >> 31) & ((POLY >> 7) ^ (POLY >> 1))) ^ (((crc << 30) >> 31) & ((POLY >> 6) ^ POLY))
        ^ (((crc << 29) >> 31) & (POLY >> 5)) ^ (((crc << 28) >> 31) & (POLY >> 4))
        ^ (((crc << 27) >> 31) & (POLY >> 3)) ^ (((crc << 26) >> 31) & (POLY >> 2))
        ^ (((crc << 25) >> 31) & (POLY >> 1)) ^ (((crc << 24) >> 31) & POLY);
    crc = (crc >> 8) ^ c;
  }
  *resultCrc32 = crc;
}

__host__ void my::cuda::JAMCRC_byte_tableless2(
    const dim3& grid, const dim3& threads, uchar* data, ulong length, uint previousCrc32, uint* resultCrc32)
{
  JAMCRC_byte_tableless2_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

__host__ void my::cuda::JAMCRC_byte_tableless2(const dim3& grid,
                                               const dim3& threads,
                                               cudaStream_t& stream,
                                               uchar* data,
                                               ulong length,
                                               uint previousCrc32,
                                               uint* resultCrc32)
{
  JAMCRC_byte_tableless2_kernel<<<grid, threads, 0, stream>>>(data, length, previousCrc32, resultCrc32);
}

extern "C" __host__ void JAMCRC_byte_tableless2(
    const dim3& grid, const dim3& threads, uchar* data, ulong length, uint previousCrc32, uint* resultCrc32)
{
  JAMCRC_byte_tableless2_kernel<<<grid, threads>>>(data, length, previousCrc32, resultCrc32);
}

__global__ void vecMult_kernel(int* a, int* b, int* c, size_t n)
{
  // Get our global thread ID
  int id = blockIdx.x * blockDim.x + threadIdx.x;

  // Make sure we do not go out of bounds
  if (id < n)
    c[id] = a[id] * b[id];
}

__host__ void my::cuda::vecMult(size_t gridSize, size_t blockSize, int* a, int* b, int* c, size_t n)
{
  vecMult_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__host__ void my::cuda::vecMult(
    size_t gridSize, size_t blockSize, cudaStream_t& stream, int* a, int* b, int* c, size_t n)
{
  vecMult_kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, n);
}

extern "C" __host__ void vecMult(size_t gridSize, size_t blockSize, int* a, int* b, int* c, size_t n)
{
  vecMult_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__global__ void matrixMultiplySimple_kernel(int* a, int* b, int* c, size_t width)
{
  size_t col = threadIdx.x + blockIdx.x * blockDim.x;
  size_t row = threadIdx.y + blockIdx.y * blockDim.y;

  int result = 0;

  if (col < width && row < width) {
    for (size_t k = 0; k < width; k++) {
      result += a[row * width + k] * b[k * width + col];
    }
    c[row * width + col] = result;
  }
}

__host__ void my::cuda::matrixMultiplySimple(dim3 gridSize, dim3 blockSize, int* a, int* b, int* c, size_t n)
{
  matrixMultiplySimple_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}

__host__ void my::cuda::matrixMultiplySimple(
    dim3 gridSize, dim3 blockSize, cudaStream_t& stream, int* a, int* b, int* c, size_t n)
{
  matrixMultiplySimple_kernel<<<gridSize, blockSize, 0, stream>>>(a, b, c, n);
}

extern "C" __host__ void matrixMultiplySimple(dim3 gridSize, dim3 blockSize, int* a, int* b, int* c, size_t n)
{
  matrixMultiplySimple_kernel<<<gridSize, blockSize>>>(a, b, c, n);
}
