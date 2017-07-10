#include <ComputeDistance.h>
#include <YotoFace.h>
using namespace std;

#define CUBLAS 0

float dotProduct(const vector<float>& v1, const vector<float>& v2)
 {
        assert(v1.size() == v2.size());
        float ret = 0.0;
        for (vector<float>::size_type i = 0; i != v1.size(); ++i)
         {
                ret += v1[i] * v2[i];
         }
        return ret;
 }
float module(const vector<float>& v)
 {
        float ret = 0.0;
        for (vector<float>::size_type i = 0; i != v.size(); ++i)
             {
                ret += v[i] * v[i];
             }
        return sqrt(ret);
}
float cosine(const vector<float>& v1, const vector<float>& v2)
{
   assert(v1.size() == v2.size());
   return dotProduct(v1, v2) / (module(v1) * module(v2));
}


int main(int argc, char*argv[])
{  
    Mat lena = imread(argv[1]);
    Mat test = imread(argv[2]);
    cout<<"YotoFace..."<<endl;
    YotoFace Yoto;
    clock_t t1, t2;
    t1 = clock();
    Yoto.Compare(lena, test);
    cout << "计算耗时" << clock() - t1 << "ms" << endl;
    //imshow("LENA", lena);
    //imshow("TEST", test);
    //waitKey(0);


    return 0;
}
