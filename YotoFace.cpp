#include <YotoFace.h>
#include"FaceProcessing.h"
#include"Utils.h"

namespace caffe
{
    extern INSTANTIATE_CLASS(InputLayer);
    extern INSTANTIATE_CLASS(InnerProductLayer);
    extern INSTANTIATE_CLASS(DropoutLayer);
    //extern INSTANTIATE_CLASS(ConvolutionLayer);
    //REGISTER_LAYER_CLASS(Convolution);
    //extern INSTANTIATE_CLASS(ReLULayer);
    //REGISTER_LAYER_CLASS(ReLU);
    //extern INSTANTIATE_CLASS(PoolingLayer);
    //REGISTER_LAYER_CLASS(Pooling);
    //extern INSTANTIATE_CLASS(LRNLayer);
    //REGISTER_LAYER_CLASS(LRN);
    //extern INSTANTIATE_CLASS(SoftmaxLayer);
    //REGISTER_LAYER_CLASS(Softmax);
    //extern INSTANTIATE_CLASS(MemoryDataLayer);
}

//构造
YotoFace::YotoFace()
{
    net = new caffe::Net<float>("model/vgg_extract_feature_memorydata.prototxt", caffe::TEST);
    net->CopyTrainedLayersFrom("model/VGG_FACE.caffemodel");
    memory_layer = (caffe::MemoryDataLayer<float> *)net->layers()[0].get();
    
    dlib::deserialize("model/shape_predictor_68_face_landmarks.dat") >> sp;//读入标记点文件
}

//提取特征
vector<float> YotoFace::ExtractFeature(Mat img_224) //ensure input 224*224!!!
{
    std::vector<Mat> test{ img_224 };
    std::vector<int> testLabel{ 0 };
    memory_layer->AddMatVector(test, testLabel);// memory_layer and net , must be define be a global variable.
    vector<caffe::Blob<float>*> input_vec;
    net->Forward(input_vec);
    auto fc7 = net->blob_by_name("fc7");//提取fc7层！4096维特征
    float* begin = fc7->mutable_cpu_data();
    vector<float> feature{ begin, begin + fc7->channels() };
    //cout << fc7->channels();
    return move(feature);
}

bool YotoFace::FaceDetect(const Mat& aImg, vector<Mat>&vecFaces, vector<cv::Rect> &vecRect)
{
     // Turn OpenCV's Mat into something dlib can deal with.  Note that this just
     // wraps the Mat object, it doesn't copy anything.  So cimg is only valid as
     // long as temp is valid.  Also don't do anything to temp that would cause it
     // to reallocate the memory which stores the image as that will make cimg
     // contain dangling pointers.  This basically means you shouldn't modify temp
     // while using cimg.
     dlib::cv_image<dlib::bgr_pixel> lImg(aImg);
     dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();
     // Detect faces 
     std::vector<dlib::rectangle> faces = detector(lImg);
     // Find the pose of each face.
     std::vector<dlib::full_object_detection> shapes;

    Mat gray,error;
    cvtColor(aImg, gray, CV_BGR2GRAY);
    
    cout<<"detect faces:"<<faces.size()<<endl;
    for (unsigned long i = 0; i < faces.size(); ++i)
    {
        dlib::full_object_detection shape = sp(dlib::cv_image<uchar>(gray), faces[i]);//标记点
        std::vector<dlib::full_object_detection> shapes;
        shapes.push_back(shape);//把点保存在了shape中
        dlib::array<dlib::array2d<dlib::rgb_pixel>>  face_chips;
        dlib::extract_image_chips(dlib::cv_image<uchar>(gray), dlib::get_face_chip_details(shapes), face_chips);
        Mat pic = dlib::toMat(face_chips[0]);
        //cvtColor(pic, pic, CV_BGR2GRAY);
        resize(pic, pic, Size(224, 224));
    
	imshow("detect face", pic);
	waitKey(10);
        //vecFaces.push_back(FaceProcessing(pic));
        vecFaces.push_back(pic);
	vecRect.push_back(dlibRectangleToOpenCV(faces[i]));

    }
    if(faces.size() > 0 )
	return true;
    return false;//如果没有检测出人脸 将返回一个空矩阵
}

//单张图片生成SingleFace
bool YotoFace::Generate(Mat input)
{
    vector<Rect> vecRoi;
    vector<Mat> vecImg224;
    if (FaceDetect(input,vecImg224, vecRoi))
    {
        int i = 0;
        for(auto img:vecImg224)
	{
            SingleFace singleface;
            //resize(img_224, img_224, Size(224, 224));
            auto feature=ExtractFeature(img);
            if (!feature.empty())
            {
                singleface.sourceImage = input;
                singleface.position = vecRoi[i++];
                singleface.feature = feature;
                singleface.roi224 = img;
            
                FaceArray.push_back(singleface);
            }
	}
    }
    else
    {
        return false;
    }
}

bool YotoFace::Generate(const Mat& input, vector<Rect>&vecRoi, vector<SingleFace> &vecFace)
{
    vector<Mat> vecImg224;
    if (FaceDetect(input,vecImg224, vecRoi))
    {
        int i = 0;
        for(auto img:vecImg224)
	{
            SingleFace singleface;
            //resize(img_224, img_224, Size(224, 224));
            auto feature=ExtractFeature(img);
            if (!feature.empty())
            {
                singleface.sourceImage = input;
                singleface.position = vecRoi[i++];
                singleface.feature = feature;
                singleface.roi224 = img;
            
                vecFace.push_back(singleface);
            }
	}
	if(vecFace.size()>0)
	   return true;
    }
    else
    {
        return false;
    }
}

void YotoFace::drawFaceImage(const Mat& input, vector<Rect>&vecRec, Mat&draw)
{
     for(auto&rec:vecRec)
     {
        rectangle(draw, rec, Scalar(0, 0, 255), 2);
     }
}

inline double LikeValue(float *v1, float *v2, int channels)
{
    //计算内积：
    register double mult = 0;
    register double v1_2 = 0;
    register double v2_2 = 0;
    for (int i = 0; i < channels; i++)
    {
        mult += v1[i] * v2[i];
        v1_2 += pow(v1[i], 2);
        v2_2 += pow(v2[i], 2);
    }

    return mult / (sqrt(v1_2)*sqrt(v2_2));
}


int YotoFace::Compare(const Mat &img1, const Mat& img2 )
{
    //解析：
    vector<SingleFace> vecFace1;
    vector<SingleFace> vecFace2;
    vector<Rect>vecRect1;
    vector<Rect>vecRect2;
    if (Generate(img1, vecRect1,vecFace1))
    {
       Mat draw;
       drawFaceImage(img1, vecRec1, draw);
    }
    if (Generate(img2, vecRect2, vecFace2))
    {
    }

    for(auto face1:vecFace1)
    {
        for(auto face2:vecFace2)
	{
            int single_channel = face1.feature.size();
            int single_channel1 = face2.feature.size();
            assert(single_channel == single_channel);
            float *faces_feature1 = &face1.feature[0];
            float *faces_feature2 = &face2.feature[0];
            float cos = LikeValue(faces_feature1, faces_feature2, single_channel);

            LOG(INFO)<< "余弦距离为：" << cos << endl;
	}
    }
    
    return 0;
}

SingleFace YotoFace::Recognition(Mat input_, SingleFace &singleface)
{
    //解析：
    if (Generate(input_))
    {
        float *single_feature = &singleface.feature[0];
        int single_channel = singleface.feature.size();

        int size_ = FaceArray.size();//有多少个人脸需要对比的
        vector<double> like_array;
        for (int i = 0; i < size_; i++)
        {
            float *faces_feature = &FaceArray[i].feature[0];
            like_array.push_back(LikeValue(single_feature, faces_feature, single_channel));
        }

        vector<double>::iterator biggest = std::max_element(std::begin(like_array), std::end(like_array));
        int max_ = distance(std::begin(like_array), biggest);
        LOG(INFO) << "余弦距离为cosine：" << max_ << endl;

        return FaceArray[max_];

    }

    else
    {
        return singleface;
    }

}

