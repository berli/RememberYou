#include <Register.h>
#include <FaceDetect.h>
#include <ComputeDistance.h>
#include <ExtractFeature_.h>

float* Register::MatToVector_(Mat TrainMat)
{
    Mat TrainMat_ROI = Facedetect(TrainMat);
    if (!TrainMat_ROI.empty())
        return ExtractFeature_(TrainMat_ROI);
    else return NULL;
}

void arrayJoinArray2d(float *feature, float *FaceMatrix[], int i)//实现人脸向量加入人脸矩阵
{
    FaceMatrix[i] = feature;
}
void Register::JoinFaceSpace_(Mat newFace, string FaceSpace, string Name)
{
    float *FaceVector = MatToVector_(newFace);
    if (FaceVector!=NULL)//如果不为空，即存在人脸
    {
        //加入两个数据表
        arrayJoinArray2d(FaceVector, FaceMatrix, FaceName.size());
        FaceName.push_back(Name);
        //保存这两个数据表
        //格式为：矩阵类型；向量类型
        //下次可直接读入
        SaveVector_(FaceMatrix, FaceName, FaceSpace);
    }
    else
    {
        cout << "Please try again,We can not find your face." << endl;
    }
}
float** Register::LoadVector_(string FaceSpace) // start load the features.
{

    string FaceVectorRoad = "data/" + FaceSpace + "_FaceMatrix.xml";
    string NameVectorRoad = "data/" + FaceSpace + "_NameVector.txt";
    vector<string>   NameVector;
    NameVector=LoadNameVector(NameVector, NameVectorRoad);
    if ( !NameVector.empty())
    {
        FaceName = NameVector;
        FaceMatrix_mat = LoadMat(FaceVectorRoad);
        if (!FaceMatrix_mat.empty())
        {
            float** a = MatToVector2d(FaceMatrix_mat);
            cout << "Sucessfully read the FaceSpace:" + FaceSpace + "'s data!" << endl;
            return a;
        }
    }
    else { cout << "There is no data in this FaceSpace:" + FaceSpace + ",Please input ." << endl; }
}
