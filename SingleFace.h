#pragma once
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

class SingleFace
{
public:
    string label;//人的名字
    Mat sourceImage;//原图
    Mat roi224;//截取人脸后224*224的图像
    Rect position;//其在原图中的位置
    vector<float> feature;//人脸ROI提取出的向量

    bool empty() //根据224*224图像判断类是否为空
    {
        if (roi224.empty())
            return true;
        else
            return false;
    }

    void draw()//画画
    {
        rectangle(sourceImage, position, Scalar(0, 0, 255));
    }
private:
};
