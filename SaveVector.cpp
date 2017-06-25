#include <SaveVector.h>
Mat Vector2dToMat(float **FaceMatrix,int rows)
{
    //know FaceMatrix's col and row.
    //FaceVector->Mat
    Mat T(rows, 2622, CV_32F);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < 2622; j++)
        {
            T.at<float>(i, j) = FaceMatrix[i][j];
        }
    return T;
}
void SaveMat(Mat &FaceMatrix_,string filename)
{
    FileStorage fs(filename, FileStorage::WRITE);
    fs << "FaceMatrix" << FaceMatrix_;
    fs.release();
}
Mat LoadMat(string file)//文件名
{
    FileStorage fs(file, FileStorage::READ);
    Mat FaceMatrix_;
    fs["FaceMatrix"] >> FaceMatrix_;
    return FaceMatrix_;
}
float** MatToVector2d(Mat &FaceMatrix_mat)
{
    float **array2d = new float*[FaceMatrix_mat.rows];
    for (int i = 0; i<FaceMatrix_mat.rows; ++i)
        array2d[i] = new float[FaceMatrix_mat.cols];

    for (int i = 0; i<FaceMatrix_mat.rows; ++i)
        array2d[i] = FaceMatrix_mat.ptr<float>(i);

    return array2d;
}

void SaveFaceMatrix(float *FaceMatrix[], string filename,int rows)
{
    Mat T = Vector2dToMat(FaceMatrix, rows);
    if (!T.empty())
        SaveMat(T, filename);
    else 
    { 
        cout << "Please check out your the matrix and the file.We can not read any information." << endl;
        exit(0);
    }
}

//存储姓名
void SaveNameVector(vector<string>   &NameVector, string filename){
    int Num = 0;
    ofstream NameSaveFile(filename, ios::app);
    while (Num < NameVector.size())
        NameSaveFile << NameVector[Num++] << endl;
    NameSaveFile.clear();
}
vector<string> LoadNameVector(vector<string>   &NameVector_, string filename)
{
    ifstream in(filename);
    int Num = 0;
    string line;
    if (in){
        while (getline(in, line))
        {
            NameVector_.push_back(line);
        }
    }
    in.clear();
    return NameVector_;
}
