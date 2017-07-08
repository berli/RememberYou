#include <FaceDetect.h>
#include <FaceRotate.h>
#include <FaceProcessing.h>
#include <FaceProcessing.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

using namespace dlib;

shape_predictor sp;

void Dlib_Prodefine()
{
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;//读入标记点文件
}
Mat FaceDetect(Mat frame)//脸是否存在
{
    Mat gray,error;
    cvtColor(frame, gray, CV_BGR2GRAY);
    int * pResults = NULL;
///    pResults = facedetect_frontal_tmp((unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, gray.step, 1.2f, 5, 24);
    int peopleNUM = (pResults ? *pResults : 0);

    for (int i = 0; i < peopleNUM; i++)//代表有几张人脸(pResults ? *pResults : 0)
    {
        short * p = ((short*)(pResults + 1)) + 6 * i;
        Rect opencvRect(p[0], p[1], p[2], p[3]);
        //gray = gray(opencvRect);
        dlib::rectangle dlibRect((long)opencvRect.tl().x, (long)opencvRect.tl().y, (long)opencvRect.br().x - 1, (long)opencvRect.br().y - 1);
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(gray), dlibRect);//标记点
        std::vector<full_object_detection> shapes;
        shapes.push_back(shape);//把点保存在了shape中
        dlib::array<array2d<rgb_pixel>>  face_chips;
        extract_image_chips(dlib::cv_image<uchar>(gray), get_face_chip_details(shapes), face_chips);
        Mat pic = toMat(face_chips[0]);
        cvtColor(pic, pic, CV_BGR2GRAY);
        resize(pic, pic, Size(224, 224));
        return FaceProcessing(pic);
    }
    return error;//如果没有检测出人脸 将返回一个空矩阵
}
