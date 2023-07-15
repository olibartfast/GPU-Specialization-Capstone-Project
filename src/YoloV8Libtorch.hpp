#pragma once
#include "common.hpp"
#include <torch/torch.h>
#include <torch/script.h>

class YoloV8Libtorch : public YoloV8
{
private:
    torch::DeviceType device_;
    torch::jit::script::Module module_;

public:

    YoloV8Libtorch(const std::string& model_path, bool use_gpu = false)
    {
        logger_->info("Initializing YoloV8Libtorch");
        if (use_gpu && torch::cuda::is_available())
        {
            device_ = torch::kCUDA;
            logger_->info("Using CUDA GPU");
        }
        else
        {
            device_ = torch::kCPU;
            logger_->info("Using CPU");
        }

        module_ = torch::jit::load(model_path, device_);

        // In libtorch i didn't find a way to get the input layer
        input_width_ = 640;
        input_height_=640;
        channels_=3;

    }

    std::tuple<std::vector<Detection>, Mask>  infer(const cv::Mat& image) override
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
        torch::Tensor output_tensor0 = output.toTuple()->elements()[0].toTensor();
        torch::Tensor output_tensor1 = output.toTuple()->elements()[1].toTensor();

        // Convert the output tensors to CPU and extract data
        output_tensor0 = output_tensor0.to(torch::kCPU).contiguous();
        output_tensor1 = output_tensor1.to(torch::kCPU).contiguous();
        const float* output0 = output_tensor0.data_ptr<float>();
        const float* output1 = output_tensor1.data_ptr<float>();
    
        // Get the shapes of the output tensors
        std::vector<int64_t> shape0 = output_tensor0.sizes().vec();
        std::vector<int64_t> shape1 = output_tensor1.sizes().vec();
        cv::Size frame_size(image.cols, image.rows);
        return postprocess(output0, output1, shape0, shape1, frame_size);

    }
};
