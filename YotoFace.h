#pragma once
#include <SingleFace.h>
#include <caffe/caffe.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include "caffe/layers/input_layer.hpp"  
#include "caffe/layers/inner_product_layer.hpp" 
#include "caffe/layers/dropout_layer.hpp"  
#include "caffe/layers/conv_layer.hpp"  
#include "caffe/layers/relu_layer.hpp"  
#include <caffe/layers/memory_data_layer.hpp>
#include "caffe/layers/pooling_layer.hpp" 
#include "caffe/layers/lrn_layer.hpp"  
#include "caffe/layers/softmax_layer.hpp"  
#include <caffe/proto/caffe.pb.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/opencv.h>

#include <mutex>

class YotoFace
{
public:
    YotoFace();
    vector<float> ExtractFeature(Mat input_224);//提取特征
    bool Generate(Mat input);//没有label的情况下进行的Generate，表示的是识别
    bool Generate(const Mat& input, vector<Rect>&vecRoi,vector<SingleFace> &vecFace);
   int Compare(const Mat &img1, const Mat& img2 );

    SingleFace Recognition(Mat input_224, SingleFace &singleface);//输入一个224*224的图片，查找他属于哪个人
    vector<SingleFace> FaceArray;//众多的SingleFace

    //仅仅用于人脸检测
    void drawFaceImage(const Mat& input, vector<Rect>&vecRec, Mat&draw);
private:
    caffe::MemoryDataLayer<float> *memory_layer;//进行数据输入的层
    caffe::Net<float>* net;//整个layer和权重
    bool FaceDetect(const Mat& aImg, vector<Mat>&vecFaces, vector<cv::Rect> &vecRect);
    
    dlib::shape_predictor sp;

    string csDlibMode;
};
