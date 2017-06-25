#ifndef CAFFE_NET_MEMORYLAYER_H_H
#define CAFFE_NET_MEMORYLAYER_H_H
#include "caffe/layers/input_layer.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/layers/dropout_layer.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/relu_layer.hpp"
#include <iostream>
#include "caffe/caffe.hpp"
#include <opencv.hpp>
#include <caffe/layers/memory_data_layer.hpp>
#include "caffe/layers/pooling_layer.hpp"
#include "caffe/layers/lrn_layer.hpp"
#include "caffe/layers/softmax_layer.hpp"
// must predefined
caffe::MemoryDataLayer<float> *memory_layer;
caffe::Net<float>* net;
#endif// CAFFE_NET_MEMORYLAYER_H_H
