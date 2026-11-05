cmake_minimum_required(VERSION 3.16)
project(cuda_kernels_lab_bench_tool CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(bench_tool bench_tool.cpp)
target_compile_options(bench_tool PRIVATE -Wall -Wextra)
