#include <ExtractFeature_.h>

#ifdef EXTRACT
int main()
{
    Caffe_Predefine();
    Mat lena = imread("lena.jpg");
    if (!lena.empty())
    {
        int i = 0;
        vector<float> print=ExtractFeature(lena);
        while (i<print.size())
        {
            cout << print[i++] << endl;
        }
    }
    imshow("Extract feature",lena);
    waitKey(0);
}
#endif

