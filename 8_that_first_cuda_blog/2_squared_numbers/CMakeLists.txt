cmake_minimum_required(VERSION 3.18)
project(squared_numbers LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(squared_numbers squared_numbers.cu)