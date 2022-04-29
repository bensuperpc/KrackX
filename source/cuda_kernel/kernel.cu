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

#include "cuda_kernel/kernel.cuhpp"

__global__ void runner_kernel(uint32_t* crc_result, uint64_t* index_result, uint64_t array_size, uint64_t a, uint64_t b)
{
  uint64_t id = blockIdx.x * blockDim.x + threadIdx.x + a;
  if (id <= b && id >= a) {
    // printf("blockIdx %d, blockDim %d, threadIdx %d\n", blockIdx.x, blockDim.x, threadIdx.x);

    // Allocate memory for the array
    unsigned char array[29] = {0};

    uint64_t size = 0;
    // Generate the array
    find_string_inv_kernel(array, id, size);

    uint32_t result = 0;

    // Calculate the CRC
    jamcrc_kernel(array, result, size, 0);

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

__host__ void my::cuda::launch_kernel(size_t gridSize,
                                      size_t blockSize,
                                      cudaStream_t& stream,
                                      uint32_t* crc_result,
                                      uint64_t* index_result,
                                      uint64_t array_size,
                                      uint64_t a,
                                      uint64_t b)
{
  runner_kernel<<<gridSize, blockSize, 0, stream>>>(crc_result, index_result, array_size, a, b);
}

__device__ void find_string_inv_kernel(unsigned char* array, uint64_t n, uint64_t& terminator_index)
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

__device__ void jamcrc_kernel(const void* data, uint32_t& result, uint64_t length, uint32_t previousCrc32)
{
  uint32_t crc = ~previousCrc32;
  unsigned char* current = (unsigned char*)data;
  while (length--)
    crc = (crc >> 8) ^ crc32_lookup[(crc & 0xFF) ^ *current++];
  result = crc;
}