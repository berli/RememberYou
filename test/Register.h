#include <opencv2/opencv.hpp>
#include <SaveVector.h>
using namespace cv;
using namespace std;

class Register
{
public:
    string FaceSpace;//The name of FaceSpace
    vector<string> FaceName;//People's name ,the same as FaceNumber
    //float *
    float* MatToVector_(Mat TrainMat);//将Mat在人脸识别、预处理后转换为一个向量
    float *FaceMatrix[20];//20个人
    void JoinFaceSpace_(Mat newFace, string FaceSpace, string FaceName);//join the new face to FaceSpace
    float** LoadVector_(string FaceSpace);//读入数据
    Mat FaceMatrix_mat;//临时存储读取的Mat类型
private:

    void SaveVector_(float *FaceMatrix_[], vector<string> FaceName_, string FaceSpace_) // save the people's  face vector
    {
        //使用xml来存储数据
        if (!(FaceMatrix_ == NULL) && !FaceName_.empty())
        {
            SaveFaceMatrix(FaceMatrix_, "data/" + FaceSpace_ + "_FaceMatrix.xml", FaceName_.size());
            SaveNameVector(FaceName, "data/" + FaceSpace_ + "_NameVector.txt");
        }
        else { cout << "Sorry.There are some problems while saving your face and name. please try again" << endl; }
    }
};
