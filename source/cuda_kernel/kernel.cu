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

__global__ void runner_kernel(uint32_t* crc_result, uint64_t* index_result, uint64_t array_size, uint64_t a, uint64_t b)
{
  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x + a;
  if (id < b && id >= a) {
    // printf("blockIdx %d, blockDim %d, threadIdx %d\n", blockIdx.x, blockDim.x, threadIdx.x);
    crc_result[1] = 4;
    index_result[1] = 8;

    int32_t array_size = 29;

    /*
    if (id > 26) {
      array_size = ceil(logf(id) / logf(26));
    } else {
      array_size = 1;
    }*/

    // printf("array_size %d\n", array_size);

    // Allocate memory for the array
    unsigned char* array = (unsigned char*)malloc(array_size * sizeof(unsigned char));
    if (array == NULL) {
      return;
    }
    // Generate the array
    find_string_inv_kernel(array, id);

    uint32_t* result = (uint32_t*)malloc(sizeof(uint32_t));
    if (result == NULL) {
      return;
    }
    *result = 0;
    // printf("array: %s\n", array);

    // Calculate the CRC

    int32_t size = 0;
    for (; size < array_size; size++) {
      if (array[size] == '\0') {
        break;
      }
    }
    // printf("size %d\n", size);
    jamcrc_kernel(array, result, size, 0);
    // printf("result: %u\n", *result);

    bool found = false;
    for (uint32_t i = 0; i < 87; i++) {
      if (*result == crc32_lookup[i]) {
        found = true;
        break;
      }
    }

    if (!found) {
      // printf("NOT FOUND!\n");
      return;
    } else {
      printf("FOUND!\n");
    }

    //__syncthreads();

    crc_result[0] = *result;
    index_result[0] = id;

    free(array);
    free(crc_result);
  }
}

__host__ void my::cuda::launch_kernel(size_t gridSize,
                                      size_t blockSize,
                                      cudaStream_t& stream,
                                      uint32_t* crc_result,
                                      uint64_t* index_result,
                                      uint64_t array_size,
                                      uint64_t a,
                                      uint64_t b)
{
  printf("array_size %d, a %d, b %d\n", array_size, a, b);
  runner_kernel<<<gridSize, blockSize, 0, stream>>>(crc_result, index_result, array_size, a, b);
}

__device__ void find_string_inv_kernel(uchar* array, uint64_t n)
{
  const uint32_t string_size_alphabet = 27;

  const uchar alpha[string_size_alphabet] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};
  // If n < 27
  if (n < 26) {
    array[0] = alpha[n];
    array[1] = '\0';
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
}

__device__ void jamcrc_kernel(const void* data, uint32_t* result, uint64_t length, uint32_t previousCrc32)
{
  uint32_t crc = ~previousCrc32;
  unsigned char* current = (unsigned char*)data;
  while (length--)
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  *result = crc;
}