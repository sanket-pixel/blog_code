cmake_minimum_required(VERSION 3.18)
project(hello_world LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(hello_world_cpu hello_world_cpu.cpp)

add_executable(hello_world_gpu hello_world_gpu.cu)