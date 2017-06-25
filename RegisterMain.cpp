#include <Register.h>
#include <FaceDetect.h>
#include <ExtractFeature_.h>

int main()
{
    Caffe_Predefine();
    Dlib_Predefine();
    Register train;
    Mat lena = imread("lena.jpg");
    train.JoinFaceSpace_(lena,"LLEENNAA","lena");

    cout << "当前的人脸矩阵的第一个元素为" << train.FaceMatrix[0][0] << endl;
    cout << "当前的人脸名容器的第一个元素为" << train.FaceName[0]<< endl;

    Register test;

    cout<<"读取保存的人脸矩阵，其第一个元素为"<<test.LoadVector_("LLEENNAA")[0][0]<<endl;
    cout << "读取保存的人脸名字容器，其第一个元素为" << test.FaceName[0]<<endl;
    imshow("lena.jpg",lena);
    waitKey(0);
}
