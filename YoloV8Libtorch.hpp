#pragma once
#include "common.hpp"
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/ir/ir.h>

class YoloV8Libtorch : public YoloV8
{
private:
    torch::DeviceType device_;
    torch::jit::script::Module module_;

public:
    std::string print_shape(const std::vector<std::int64_t>& v)
    {
        // Implementation of print_shape function...
    }

    YoloV8Libtorch(const std::string& model_path, bool use_gpu = false)
    {
        if (use_gpu && torch::cuda::is_available())
        {
            device_ = torch::kCUDA;
            std::cout << "Using CUDA GPU" << std::endl;
        }
        else
        {
            device_ = torch::kCPU;
            std::cout << "Using CPU" << std::endl;
        }

        module_ = torch::jit::load(model_path, device_);

        // In libtorch i didn't find a way to get the input layer
        input_width_ = 640;
        input_height_=640;
        channels_=3;

    }

    void infer(const cv::Mat& image) override
    {
        // Preprocess the input image
        std::vector<float> input_tensor = preprocess_image(image);

        // Convert the input tensor to a Torch tensor
        torch::Tensor input = torch::from_blob(input_tensor.data(), { 1, channels_, input_height_, input_width_ }, torch::kFloat32);
        input = input.to(device_);

        // Run inference
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        auto output = module_.forward(inputs);
        torch::Tensor t0 = output.toTuple()->elements()[0].toTensor();
        torch::Tensor t1 = output.toTuple()->elements()[1].toTensor();

        // Print the shapes
        std::cout << "t0 shape: " << t0.sizes() << std::endl;
        std::cout << "t1 shape: " << t1.sizes() << std::endl;

        // Postprocess the output tensor
        // Implementation of postprocessing...

        // Print the shape of the output tensor
        // std::cout << "Output tensor shape: " << print_shape(output.sizes().vec()) << std::endl;

    }
};
