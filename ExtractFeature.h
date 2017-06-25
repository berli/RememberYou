 #include <opencv.hpp>
using namespace cv;
using namespace std;

float* ExtractFeature_(Mat FaceROI);//添加一个提取特征的函数
vector<float> ExtractFeature(Mat FaceROI);
void Caffe_Predefine();
