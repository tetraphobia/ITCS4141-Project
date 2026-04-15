/**
 From https://docs.opencv.org/3.4.0/d3/dc1/tutorial_basic_linear_transform.html
 Check the webpage for description
 */
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
using namespace std;
using namespace cv;

/// Epic filter masks
short lpf_filter_6[3][3] = {{0, 1, 0}, {1, 2, 1}, {0, 1, 0}};
short lpf_filter_9[3][3] = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
short lpf_filter_10[3][3] = {{1, 1, 1}, {1, 2, 1}, {1, 1, 1}};
short lpf_filter_16[3][3] = {{1, 2, 1}, {2, 4, 2}, {1, 2, 1}};
short lpf_filter_32[3][3] = {{1, 4, 1}, {4, 12, 4}, {1, 4, 1}};
short hpf_filter_1[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
short hpf_filter_2[3][3] = {{-1, -1, -1}, {-1, 9, -1}, {-1, -1, -1}};
short hpf_filter_3[3][3] = {{1, -2, 1}, {-2, 5, -2}, {1, -2, 1}};

/// EPIC implementation
void filter_image(Mat &input_img, Mat &output_img, short filter[3][3],
                  int divisor) {
  int channels = input_img.channels();

  for (int i = 1; i < input_img.rows - 1; i++) {
    for (int j = 1; j < input_img.cols - 1; j++) {
      for (int c = 0; c < channels; c++) {
        int sum = 0;

        for (int a = -1; a < 2; a++) {
          for (int b = -1; b < 2; b++) {
            sum += input_img.at<Vec3b>(i + a, j + b)[c] * filter[a + 1][b + 1];
          }
        }

        sum /= divisor;
        sum = std::clamp(sum, 0, 255);

        output_img.at<Vec3b>(i, j)[c] = saturate_cast<uchar>(sum);
      }
    }
  }
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
  // for (int y = 0; y < image.rows; y++) {
  //   for (int x = 0; x < image.cols; x++) {
  //     for (int c = 0; c < 3; c++) {
  //       new_image.at<Vec3b>(y, x)[c] =
  //           saturate_cast<uchar>(alpha * (image.at<Vec3b>(y, x)[c]) + beta);
  //     }
  //   }
  // }

  filter_image(image, new_image, hpf_filter_1, 1);

  namedWindow("Original Image", WINDOW_AUTOSIZE);
  namedWindow("New Image", WINDOW_AUTOSIZE);
  imshow("Original Image", image);
  imshow("New Image", new_image);
  waitKey();
  return 0;
}
