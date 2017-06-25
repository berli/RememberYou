#ifndef FACE_PROCESSING_H_H
#define FACE_PROCESSING_H_H

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
Mat FaceProcessing(const Mat &img_, double gamma = 0.2, double sigma0 = 1, double sigma1 = -2, double mask = 0, double do_norm = 10);

#endif


