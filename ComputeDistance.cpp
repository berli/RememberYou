#include <ComputeDistance.h>
#include <ExtractFeature_.h>
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
    SingleFace face;
    YotoFace Yoto;
    Mat lena = imread(argv[1]);
    Mat test = imread(argv[2]);
    Yoto.Recognition(lena, face);
    imshow("LENA", lena);
    imshow("TEST", test);
    waitKey(0);


    return 0;
    Caffe_Predefine();
    //Mat lena = imread(argv[1]);
    //Mat test = imread(argv[2]);

    imshow("lean", lena);
    resize(lena, lena, Size(224, 224));
    resize(test, test, Size(224, 224));
    if (!lena.empty()&&!test.empty())
    {
        vector<float> lena_vector = ExtractFeature(lena);
        vector<float> test_vector = ExtractFeature(test);
        clock_t t1, t2;
        t1 = clock();
        cout << "余弦距离为：" << cosine(lena_vector, test_vector) << endl;
        t2 = clock();
        cout << "计算耗时" << t2 - t1 << "ms" << endl;
    }
    else
    {
      cout<<"File is not exist"<<endl;
      return -1;
    }

    return 0;
}
#if CUBLAS
int main()
{

    int arraySize = 10000;
    float* a = (float*)malloc(sizeof(float) * arraySize);

    float* d_a;
    cudaMalloc((void**)&d_a, sizeof(float) * arraySize);

    for (int i = 0; i<arraySize; i++)
        a[i] = 1.0f;
    cudaMemcpy(d_a, a, sizeof(float) * arraySize, cudaMemcpyHostToDevice);
    float* result = (float*)malloc(sizeof(float));
    float* a_result = (float*)malloc(sizeof(float));
    float* b_result = (float*)malloc(sizeof(float));
    ret = cublasCreate(&handle_cos);
    clock_t t1, t2;
    t1 = clock();
    cout << Cosine(d_a, d_a, result, a_result, b_result, arraySize) << endl;
    t2 = clock();
    cout << t2 - t1<<"ms"<<endl;

    //printf("\n\nCUBLAS: %.3f", *cb_result);

    cublasDestroy(handle_cos);
    cin.get();
}
#endif //CUBLAS
