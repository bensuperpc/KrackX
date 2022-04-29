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
//  Source:                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "stdio.h"

__host__ void launch_kernel(size_t gridSize,
                            size_t blockSize,
                            cudaStream_t& stream,
                            uint32_t* crc_result,
                            uint64_t* index_result,
                            uint64_t array_size,
                            uint64_t a,
                            uint64_t b);

#endif