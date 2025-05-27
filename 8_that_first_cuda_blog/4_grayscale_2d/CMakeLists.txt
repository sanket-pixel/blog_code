cmake_minimum_required(VERSION 3.18)
project(grayscale_2d LANGUAGES CXX CUDA)

find_package(OpenCV REQUIRED)

add_executable(grayscale_2d grayscale_2d.cu)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(grayscale_2d ${OpenCV_LIBS})
