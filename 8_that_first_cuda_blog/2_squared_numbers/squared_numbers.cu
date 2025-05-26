#include <iostream>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 4
#define DATA_SIZE 50

__global__ void print_square() {
  unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < DATA_SIZE) {
    printf("Thread %d: %d squared = %d\n", id, id, id * id);
  }
}

int main() {
  // compute how many blocks we need
  int blocks = (DATA_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  print_square<<<blocks, THREADS_PER_BLOCK>>>();
  cudaDeviceSynchronize();
}