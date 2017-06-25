float* ExtractFeature_(Mat FaceROI)
{
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    std::vector<Mat> test;
    std::vector<int> testLabel;
    test.push_back(FaceROI);
    testLabel.push_back(0);
    memory_layer->AddMatVector(test, testLabel);// memory_layer and net , must be define be a global variable.
    test.clear();
    testLabel.clear();
    std::vector<caffe::Blob<float>*> input_vec;
    net->Forward(input_vec);
    boost::shared_ptr<caffe::Blob<float>> fc8 = net->blob_by_name("fc8");
    int test_num = 0;
    float FaceVector[2622];
    while (test_num < 2622)
    {
        FaceVector[test_num] = (fc8->data_at(0, test_num, 1, 1));
        test_num++;
    }
    return FaceVector;
}
