#include <iostream>
#include "cxxopts.hpp"

int main(int argc, char* argv[]) {
    cxxopts::Options options("UnderwaterTrashInstanceSegmentation", "Description of your program");

    // Add command line options
    options.add_options()
        ("f,framework", "Selected framework (ONNX_RUNTIME, LIBTORCH, or TENSORRT)", cxxopts::value<std::string>())
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

    std::string framework = result["framework"].as<std::string>();

    // Use the selected framework here
    if (framework == "ONNX_RUNTIME") {
        // Perform inference using ONNX Runtime
        std::cout << "Performing inference using ONNX Runtime..." << std::endl;
    }
    else if (framework == "LIBTORCH") {
        // Perform inference using LibTorch
        std::cout << "Performing inference using LibTorch..." << std::endl;
    }
    else if (framework == "TENSORRT") {
        // Perform inference using TensorRT
        std::cout << "Performing inference using TensorRT..." << std::endl;
    }
    else {
        std::cerr << "Invalid framework specified. Please use ONNX_RUNTIME, LIBTORCH, or TENSORRT." << std::endl;
        return 1;
    }

    // Rest of your code for inference

    return 0;
}
