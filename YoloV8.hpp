#pragma once
#include "common.hpp"


struct Mask{
    std::vector<size_t> segShapes;
    std::vector<float> raw_mask_protos;
    std::vector<std::vector<float>> raw_mask_proposals;
};

struct Detection{
    cv::Rect bbox;
    float score;
    float label_id;
};

class YoloV8 {
protected:



public:
    virtual void infer(const cv::Mat& image) = 0;
};