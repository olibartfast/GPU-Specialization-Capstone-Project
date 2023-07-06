#include <iostream>
#include "cxxopts.hpp"
#include "YoloV8ONNX.hpp"

std::unique_ptr<YoloV8> createYoloV8(const std::string& framework, const std::string& weights) {
    if (framework == "ONNX_RUNTIME") {
        return std::make_unique<YoloV8ONNX>(weights);
    } else if (framework == "LIBTORCH") {
        return nullptr;
    } else if (framework == "TENSORRT") {
        return nullptr;
    } else {
        return nullptr;
    }
}

int main(int argc, char* argv[]) {
    cxxopts::Options options("UnderwaterTrashInstanceSegmentation", "Description of your program");

    // Add command line options
    options.add_options()
        ("f,framework", "Selected framework (ONNX_RUNTIME, LIBTORCH, or TENSORRT)", cxxopts::value<std::string>())
        ("w,weights", "Path to weights", cxxopts::value<std::string>())
        ("v,video", "Path to video source", cxxopts::value<std::string>())
        ("h,help", "Print help");

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.count("framework")) {
        std::cerr << "Framework not specified. Use the --framework option." << std::endl;
        return 1;
    }

    if (!result.count("weights")) {
        std::cerr << "Weights not specified. Use the --weights option." << std::endl;
        return 1;
    }

    if (!result.count("video")) {
        std::cerr << "video input source not specified. Use the --video option." << std::endl;
        return 1;
    }


    std::string framework = result["framework"].as<std::string>();
    std::string weights = result["weights"].as<std::string>();
    std::string video = result["video"].as<std::string>();

    std::unique_ptr<YoloV8> yolo = createYoloV8(framework,weights);
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
