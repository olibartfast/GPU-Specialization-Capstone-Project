#pragma once
#include "common.hpp"

class YoloV8 {
protected:

public:
    virtual void infer(const cv::Mat& image) = 0;
};