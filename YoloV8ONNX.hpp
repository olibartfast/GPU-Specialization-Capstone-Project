#pragma once
#include <onnxruntime_cxx_api.h>
#include "YoloV8.hpp"

class YoloV8ONNX : public YoloV8 {
private:
    Ort::Env env;
    Ort::Session session;
    // Add other necessary members and methods for ONNX Runtime inference

public:
    YoloV8ONNX(const std::string& modelPath) : env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YoloV8ONNX")), session(env, modelPath.c_str(), Ort::SessionOptions()) {
        // Initialize the ONNX Runtime session and other necessary setup
    }

    void loadModel(const std::string& modelPath) override {
        // Load the ONNX model using ONNX Runtime API
    }

    void infer(const cv::Mat& image) override {
        // Perform inference using the loaded ONNX model and ONNX Runtime API
    }
    // Add other necessary members and methods for ONNX Runtime inference
};