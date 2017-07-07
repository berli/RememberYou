#include "ExtractFeature.h"
#include "caffe_net_memorylayer.h"
extern caffe::MemoryDataLayer<float> *memory_layer;
extern caffe::Net<float>* net;

float* ExtractFeature_(Mat FaceROI)
{
    //Caffe::set_mode(caffe::Caffe::GPU);
    std::vector<Mat> test;
    std::vector<int> testLabel;
    test.push_back(FaceROI);
    testLabel.push_back(0);
    if(memory_layer)
       memory_layer->AddMatVector(test, testLabel);// memory_layer and net , must be define be a global variable.
    else
    {
       cout<<"memory_layer is not initialize"<<endl;
       return NULL;
    }
    test.clear();
    testLabel.clear();
    std::vector<caffe::Blob<float>* > input_vec;
    if(net)
       net->Forward(input_vec);
    else
    {
       cout<<"net is not initialize"<<endl;
       return NULL;
    }
    boost::shared_ptr<caffe::Blob<float> > fc8 = net->blob_by_name("fc8");
    int test_num = 0;
    float FaceVector[2622];
    while (test_num < 2622)
    {
        FaceVector[test_num] = (fc8->data_at(0, test_num, 1, 1));
        test_num++;
    }
    return FaceVector;
}
