
#include "Recognition.h"
#include "SaveVector.h"
#include "FaceDetect.h"
#include "ExtractFeature.h"
#include "ComputeDistance.h"

void Recognition::LoadVector(string FaceSpace) // save the people's  face vector
{
    string FaceVectorRoad = "data/" + FaceSpace + "_FaceMatrix.xml";
    string NameVectorRoad = "data/" + FaceSpace + "_NameVector.txt";
    vector<vector<float> >  FaceVector;
    //FaceVector = LoadFaceMatrix(FaceVectorRoad);
    NameVector=LoadNameVector(NameVector, NameVectorRoad);
    if (!FaceVector.empty() && !NameVector.empty())
    {
        FaceMat = FaceVector;
        NameVector = NameVector;
        cout << "Sucessfully read the FaceSpace:" + FaceSpace + "'s data!" << endl;
    }
    else { cout << "There is no data in this FaceSpace:" + FaceSpace + ",Please input ." << endl; }
}

string Recognition::Predict(Mat LoadGetFace)//可优化，using CUDA TO COMPUTE
{
    if (!LoadGetFace.empty())
    {
        vector<float> v = ExtractFeature(FaceDetect(LoadGetFace));
        if (!v.empty())
        {
            int ID = -1;
            float MaxCos = 0;
            for (int i = 0; i < NameVector.size(); i++)
            {
                float t_cos = cosine(v, FaceMat[i]);
                if (t_cos > MaxCos)
                {
                    ID = i;
                    MaxCos = t_cos;//update the coff
                }
            }
            return NameVector[ID];
        }
        else
        {
            cout << "The Picture does not have people's Face,Please try again." << endl;
        }
    }
    else cout << "The picture is empty.Please Check it and make sure." << endl; 

}

void Recognition::LoadRecognitionModel(vector<vector<float> > FaceMatrix, vector<string> NameVector_)//创建识别模型，需要输入FaceMarix，NameVector.
{
    if (!FaceMatrix.empty() &&!NameVector_.empty())
    {
        FaceMat = FaceMatrix;
        NameVector = NameVector_;
    }
    else
    {
        cout << "Please check your FaceMatrix and NameVector.It may be empty." << endl;
    }
}

void Recognition::clear()
{
    //FaceMatrix_ = NULL;
    NameVector.clear();
}

void Recognition::update(vector<vector<float> > FaceMatrix, vector<string> NameVector)
{
    //clear_();
    LoadRecognitionModel(FaceMat, NameVector);
}
