#include <cuda.h>
#include <opencv2/opencv.hpp>

#define BLOCKSIZE_X 32
#define BLOCKSIZE_Y 32

using namespace std;

__global__ void convert_rgb_to_grayscale(float *dsample_image,
                                         float *dgrayscale_sample_image,
                                         int rows, int cols) {
  int global_x = blockIdx.x * blockDim.x + threadIdx.x;
  int global_y = blockIdx.y * blockDim.y + threadIdx.y;
  if (global_x < cols && global_y < rows) {
    int global_id = global_y * cols + global_x;
    float r = dsample_image[3 * global_id];
    float g = dsample_image[3 * global_id + 1];
    float b = dsample_image[3 * global_id + 2];
    dgrayscale_sample_image[global_id] = 0.144 * r + 0.587 * g + 0.299 * b;
  }
}

int main() {
  // read image from filepath
  string sample_image_path = "../sample.png";
  cv::Mat sample_image = cv::imread(sample_image_path);
  int width = sample_image.cols;
  int height = sample_image.rows;
  int channels = sample_image.channels();
  sample_image.convertTo(sample_image, CV_32F, 1.0 / 255.0);
  int sample_image_size_in_bytes = width * height * channels * sizeof(float);
  float *dsample_image;
  // allocate memory on GPU
  cudaMalloc(&dsample_image, sample_image_size_in_bytes);
  // copy image from CPU to GPU
  cudaMemcpy(dsample_image, sample_image.data, sample_image_size_in_bytes,
             cudaMemcpyHostToDevice);
  // compute number of blocks in x and y dimensions
  int number_of_blocks_x = (width + BLOCKSIZE_X - 1) / BLOCKSIZE_X;
  int number_of_blocks_y = (height + BLOCKSIZE_Y - 1) / BLOCKSIZE_Y;
  // define grid dimension and block dimension for kernel launch
  dim3 grid_dim(number_of_blocks_x, number_of_blocks_y, 1);
  dim3 block_dim(BLOCKSIZE_X, BLOCKSIZE_Y, 1);
  // allocate memory on GPU to store the grayscale image
  float *dgrayscale_sample_image;
  cudaMalloc(&dgrayscale_sample_image, width * height * sizeof(float));
  // launch the kernel
  convert_rgb_to_grayscale<<<grid_dim, block_dim>>>(
      dsample_image, dgrayscale_sample_image, height, width);
  // copy the grayscale image back from GPU to CPU
  cv::Mat himage_grayscale(height, width, CV_32FC1);
  float *himage_grayscale_data =
      reinterpret_cast<float *>(himage_grayscale.data);
  cudaMemcpy(himage_grayscale_data, dgrayscale_sample_image,
             width * height * sizeof(float), cudaMemcpyDeviceToHost);
  himage_grayscale.convertTo(himage_grayscale, CV_8U, 255.0);
  cv::imwrite("../grayscale_sample.png", himage_grayscale);
  return 0;
}