cmake_minimum_required(VERSION 3.18)
project(print_2d_coordinates LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(print_2d_coordinates print_2d_coordinates.cu)