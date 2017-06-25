#ifdef RECOGNITION_H_H
#define RECOGNITION_H_H

class Recognition
{
public:

    vector<string> NameVector;

    //vector
    void LoadVector(string FaceSpace);//读入数据,保存的名称为FaceSpace_FaceVector/FaceSpace_FaceName
    void LoadRecognitionModel(vector<vector<float>> FaceMatrix, vector<string> NameVector);//创建识别模型，需要输入FaceMarix，NameVector.
    string Predict(Mat LoadGetFace);//预测
    //用法： Recognition test; test.LoadRecognitionModel();cout<<test->predict(Mat) ; test.update()
    void clear();
    void update(vector<vector<float>> FaceMatrix, vector<string> NameVector);//change .
    vector<vector<float>> FaceMat;

private:
    vector <vector<float>> ReadVector (string FaceSpace);//input FaceSpace ,read to get vector <vector<float>>
};

#endif

