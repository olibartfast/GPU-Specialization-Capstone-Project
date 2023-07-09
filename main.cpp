#include <iostream>
#include "cxxopts.hpp"
#include "common.hpp"

// Include the appropriate header based on the selected framework
#ifdef USE_ONNX_RUNTIME
#include "YoloV8ONNX.hpp"
#elif USE_LIBTORCH
#include "YoloV8Libtorch.hpp"
#elif USE_TENSORRT
#include "YoloV8TRT.hpp"
#endif

std::unique_ptr<YoloV8> createYoloV8(const std::string& weights, const bool is_gpu) {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<YoloV8ONNX>(weights, is_gpu);
#elif USE_LIBTORCH
    return std::make_unique<YoloV8Libtorch>(weights, is_gpu);
#elif USE_TENSORRT
    return std::make_unique<YoloV8TRT>(weights);
#else
    return nullptr;
#endif
}

int main(int argc, char* argv[]) {
    cxxopts::Options options("UnderwaterTrashInstanceSegmentation", "Description of your program");

    // Add command line options
    options.add_options()
        ("w,weights", "Path to weights", cxxopts::value<std::string>())
        ("v,video", "Path to video source", cxxopts::value<std::string>())
        ("gpu", "Is gpu wanted", cxxopts::value<bool>()->default_value("false"))
        ("h,help", "Print help");

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.count("weights")) {
        std::cerr << "Weights not specified. Use the --weights option." << std::endl;
        return 1;
    }

    if (!result.count("video")) {
        std::cerr << "video input source not specified. Use the --video option." << std::endl;
        return 1;
    }

    if (!result.count("gpu")) {
        std::cerr << "gpu input not specified. Use the --gpu option." << std::endl;
        return 1;
    }


    std::string weights = result["weights"].as<std::string>();
    std::string video = result["video"].as<std::string>();
    bool is_gpu = result["gpu"].as<bool>();

    std::unique_ptr<YoloV8> yolo = createYoloV8(weights,is_gpu);
    if (!yolo) {
        std::cerr << "Invalid framework specified. Supported frameworks are ONNX_RUNTIME, LIBTORCH, and TENSORRT." << std::endl;
        return 1;
    }
    cv::VideoCapture cap(video);
    cv::Mat frame;
    while(cap.read(frame))
    {
        yolo->infer(frame);

    }


    return 0;
}
