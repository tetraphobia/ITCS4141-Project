#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

using namespace std;
using namespace cv;

// i wanna flatten these this time so that they can be accessed all row-major
// like
short lpf_filter_6[9] = {0, 1, 0, 1, 2, 1, 0, 1, 0};
short lpf_filter_9[9] = {1, 1, 1, 1, 1, 1, 1, 1, 1};
short lpf_filter_10[9] = {1, 1, 1, 1, 2, 1, 1, 1, 1};
short lpf_filter_16[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
short lpf_filter_32[9] = {1, 4, 1, 4, 12, 4, 1, 4, 1};
short hpf_filter_1[9] = {0, -1, 0, -1, 5, -1, 0, -1, 0};
short hpf_filter_2[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
short hpf_filter_3[9] = {1, -2, 1, -2, 5, -2, 1, -2, 1};

// chosen filter will be stored in constant memory
// how have I not seen this before?
__constant__ short selected_filter[9];

#define BLOCK_W 16
#define BLOCK_H 16

__global__ void _filter_kernel(const unsigned char *in, unsigned char *out,
                               int rows, int cols, int channels, int divisor) {
  extern __shared__ unsigned char tile[];

  // pad both tiles to account for the halo
  const int TW = BLOCK_W + 2;
  const int TH = BLOCK_H + 2;

  const int tile_x = threadIdx.x;
  const int tile_y = threadIdx.y;
  const int j = blockIdx.x * BLOCK_W + tile_x;
  const int i = blockIdx.y * BLOCK_H + tile_y;

  // load tile into smem
  for (int shared_y = tile_y; shared_y < TH; shared_y += BLOCK_H) {
    for (int shared_x = tile_x; shared_x < TW; shared_x += BLOCK_W) {
      // shift for halo
      int global_y = blockIdx.y * BLOCK_H + shared_y - 1;
      int global_x = blockIdx.x * BLOCK_W + shared_x - 1;

      // clamp to edge for halo
      global_y = min(max(global_y, 0), rows - 1);
      global_x = min(max(global_x, 0), rows - 1);

      const unsigned char *src = in + (global_y * cols + global_x) * channels;
      unsigned char *dest = tile + (shared_y * TW + shared_x) * channels;

      // master chief, what are you doing in that shared memory tile?
      for (int c = 0; c < channels; c++) {
        // sir... im
        dest[c] = src[c];
      }
    }
  }

  __syncthreads();

  // indices of center pixels
  int shared_y_center = tile_y + 1;
  int shared_x_center = tile_x + 1;

  for (int c = 0; c < channels; c++) {
    int sum = 0;

    for (int a = -1; a <= 1; a++) {
      for (int b = -1; b <= 1; b++) {
        sum += tile[((shared_y_center + a) * TW + (shared_x_center + b)) *
                        channels +
                    c] *
               selected_filter[(a + 1) * 3 + (b + 1)];
      }
    }
    // not gonna bother putting another chungus for this
    sum /= divisor;
    // clamp
    sum = min(max(sum, 0), 255);

    out[(i * cols + j) * channels + c] = (unsigned char)sum;
  }
}

void amazing_filter_wrapper(const Mat &input_img, Mat &output_img,
                            const short filter[9], int divisor) {

  const int rows = input_img.rows;
  const int cols = input_img.cols;
  const int channels = input_img.channels();

  unsigned char *d_in = nullptr;
  unsigned char *d_out = nullptr;

  // allocate bytes for image
  const size_t img_bytes = (size_t)rows * cols * channels;
  cudaMalloc(&d_in, img_bytes);
  cudaMalloc(&d_out, img_bytes);

  cudaMemcpy(d_in, input_img.data, img_bytes, cudaMemcpyHostToDevice);

  // output is already filled with zeroes in main but on the off chance it isn't
  cudaMemset(d_out, 0, img_bytes);

  // copy in the selected filter to constant device memory
  cudaMemcpyToSymbol(selected_filter, filter, 9 * sizeof(short));

  dim3 block(BLOCK_W, BLOCK_H);
  dim3 grid((cols + BLOCK_W - 1) / BLOCK_W, (rows + BLOCK_H - 1) / BLOCK_H);
  size_t smem = (size_t)(BLOCK_W + 2) * (BLOCK_H + 2) * channels;

  _filter_kernel<<<grid, block, smem>>>(d_in, d_out, rows, cols, channels,
                                        divisor);
  cudaDeviceSynchronize();

  cudaMemcpy(output_img.data, d_out, img_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_out);
}

int main(int argc, char **argv) {
  double alpha = 1.0;                   /*< Simple contrast control */
  int beta = 0;                         /*< Simple brightness control */
  String imageName("../data/lena.jpg"); // by default
  if (argc > 1) {
    imageName = argv[1];
  }
  Mat image = imread(imageName);
  Mat new_image = Mat::zeros(image.size(), image.type());
  cout << " Basic Linear Transforms " << endl;
  cout << "-------------------------" << endl;
  cout << "* Enter the alpha value [1.0-3.0]: ";
  cin >> alpha;
  cout << "* Enter the beta value [0-100]: ";
  cin >> beta;

  amazing_filter_wrapper(image, new_image, hpf_filter_1, 1);

  namedWindow("Original Image", WINDOW_AUTOSIZE);
  namedWindow("New Image", WINDOW_AUTOSIZE);
  imshow("Original Image", image);
  imshow("New Image", new_image);
  waitKey();
  return 0;
}
