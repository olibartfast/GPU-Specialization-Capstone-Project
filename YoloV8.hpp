#pragma once
#include "common.hpp"

class YoloV8 {
public:
    virtual ~YoloV8() {}
    virtual void loadModel(const std::string& modelPath) = 0;
    virtual void infer(const cv::Mat& image) = 0;
};