##############################################################
#   ____                                                     #
#  | __ )  ___ _ __  ___ _   _ _ __   ___ _ __ _ __   ___    #
#  |  _ \ / _ \ '_ \/ __| | | | '_ \ / _ \ '__| '_ \ / __|   #
#  | |_) |  __/ | | \__ \ |_| | |_) |  __/ |  | |_) | (__    #
#  |____/ \___|_| |_|___/\__,_| .__/ \___|_|  | .__/ \___|   #
#                             |_|             |_|            #
##############################################################
#                                                            #
#  BenLib, 2021                                              #
#  Created: 16, March, 2021                                  #
#  Modified: 17, March, 2021                                 #
#  file: CMakeLists.txt                                      #
#  CMake                                                     #
#  Source:                                                   #
#  OS: ALL                                                   #
#  CPU: ALL                                                  #
#                                                            #
##############################################################

cmake_minimum_required(VERSION 3.14)
project(
    krackx_cuda_lib
    VERSION 0.1.0
    DESCRIPTION "Krack cuda lib"
    HOMEPAGE_URL "https://github.com/bensuperpc"
    LANGUAGES C CXX CUDA
)

find_package(CUDAToolkit 11.0)

set(SRCS
    kernel.cu
    wrapper.cu
	)
	
set(HEADERS
    kernel.cuh
    wrapper.hpp
    )

add_library(krackx_cuda_lib ${SRCS} ${HEADERS})

#if (CUDAToolkit_FOUND)
target_include_directories(krackx_cuda_lib PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(krackx_cuda_lib PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(krackx_cuda_lib PRIVATE CUDA::cudart)
#endif()

set_target_properties(krackx_cuda_lib
PROPERTIES
RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
