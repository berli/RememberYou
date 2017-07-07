#ifndef FACE_DETECT_H_H
#define FACE_DETECT_H_H
/*
 *author:libo
 *email:libo_5@163.com
 * */
#include <opencv2/opencv.hpp>
using namespace cv;
// define
 Mat FaceDetect(Mat frame);
// //dlib的配置函数 后面几章会讲
 void Dlib_Predefine();


#endif

