#include <iostream>
#include <cuda_runtime.h> // This include statement allows us to use cuda library in our code

__global__ void gpu_hello_world(){
  printf("Hello World from GPU! \n");
}

int main(){
  std::cout << "Hello World from CPU!" << std::endl;
  gpu_hello_world<<<1,1>>>();
  cudaDeviceSynchronize();
}
