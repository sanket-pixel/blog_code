#include <iostream>
#include <cuda_runtime.h>

#define WIDTH 10      // width of simulated 2D data (columns)
#define HEIGHT 5      // height of simulated 2D data (rows)

#define THREADS_X 4   // threads per block in X
#define THREADS_Y 2   // threads per block in Y

__global__ void print_2d_coordinates() {
  // Get global 2D index
  int global_x = blockIdx.x * blockDim.x +  threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;

  // Compute linear ID (row-major order)
  int global_id = global_y * WIDTH + global_x;

  if (global_x < WIDTH && global_y < HEIGHT) {
    printf("Block (%d,%d) Thread (%d,%d) → Global ID: %2d → Pixel (%d,%d)\n",
           blockIdx.y, blockIdx.x, threadIdx.y, threadIdx.x, global_id, global_y, global_x);
  }
}

int main() {
  dim3 threads_per_block(THREADS_X, THREADS_Y);

  // We cover WIDTH pixels in X direction using 1D grid of blocks
  int blocks_x = (WIDTH + THREADS_X - 1) / THREADS_X;
  int blocks_y = (HEIGHT + THREADS_Y - 1) / THREADS_Y;
  dim3 num_blocks(blocks_x, blocks_y);

  print_2d_coordinates<<<num_blocks, threads_per_block>>>();
  cudaDeviceSynchronize();
}