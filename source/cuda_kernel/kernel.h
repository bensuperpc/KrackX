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
//  Source:                                                 //
//  CPU: ALL                                                //
//                                                          //
//////////////////////////////////////////////////////////////

#ifndef _CUDA_KERNEL_H_
#define _CUDA_KERNEL_H_

#include <cuda.h>
#include <cuda_runtime.h>

#include "stdio.h"

void JAMCRC_byte_tableless2(
    const dim3& grid, const dim3& threads, unsigned char* data, ulong length, uint previousCrc32, uint* resultCrc32);
void vecMult(size_t gridSize, size_t blockSize, int* a, int* b, int* c, size_t n);
void matrixMultiplySimple(dim3 gridSize, dim3 blockSize, int* a, int* b, int* c, size_t n);

#endif