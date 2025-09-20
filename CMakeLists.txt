cmake_minimum_required(VERSION 3.16)
project(smoothlife LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_library(NVENCODEAPI_LIB nvidia-encode REQUIRED)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -Wall -Wextra -Wpedantic")

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG 3eda91b2e1ce7d569f84ba295507c4cd8fd96910 # version 3.2
)
FetchContent_MakeAvailable(argparse)

set(CUDA_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/kernels.cu")
set(FATBIN_FILE ${CMAKE_CURRENT_BINARY_DIR}/kernels.fatbin)
add_custom_command(
  OUTPUT ${FATBIN_FILE}
  COMMAND ${CMAKE_CUDA_COMPILER} -arch=native --fatbin ${CUDA_SOURCE} -o ${FATBIN_FILE}
  DEPENDS ${CUDA_SOURCE}
  COMMENT "Compiling ${CUDA_SOURCE} to fatbin"
)
add_custom_target(build_fatbin ALL DEPENDS ${FATBIN_FILE})

add_executable(smoothlife
  src/main.cc
  src/smoothlife.h src/smoothlife.cc
  src/encoder.h src/encoder.cc
  src/util.h)
target_include_directories(smoothlife
  PRIVATE ${CUDAToolkit_INCLUDE_DIRS}
  PRIVATE lib)
target_link_libraries(smoothlife
  PUBLIC CUDA::cuda_driver
  PUBLIC CUDA::cufft
  PUBLIC CUDA::cublas
  PUBLIC ${NVENCODEAPI_LIB}
  PUBLIC argparse)
target_compile_definitions(smoothlife PRIVATE FATBIN_FILE="${FATBIN_FILE}")
add_dependencies(smoothlife build_fatbin)
