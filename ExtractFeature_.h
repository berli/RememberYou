#ifndef EXTRACT_FEATURE_H_H
#define EXTRACT_FEATURE_H_H
#include <opencv.hpp>
using namespace cv;
using namespace std;

std::vector<float> ExtractFeature(Mat FaceROI);//给一个图片 返回一个vector<float>容器
void Caffe_Predefine();
#endif// EXTRACT_FEATURE_H_H
