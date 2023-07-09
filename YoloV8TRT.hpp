#pragma once
#include "common.hpp"
#include <NvInfer.h>  // for TensorRT API
#include <cuda_runtime_api.h>  // for CUDA runtime API
#include <memory>  // for smart pointers

class YoloV8TRT : public YoloV8
{
private:
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    nvinfer1::IExecutionContext* context_{nullptr};
    std::vector<void*> buffers_;
    std::vector<cv::Size> input_shapes_;

    nvinfer1::IRuntime* runtime_{nullptr};

public:
    // Constructor
    YoloV8TRT(const std::string& engine_path)
    {
        // // Create TensorRT runtime and deserialize the engine
        // runtime_ = nvinfer1::createInferRuntime(gLogger);
        // std::ifstream engine_file(engine_path, std::ios::binary);
        // if (!engine_file)
        // {
        //     throw std::runtime_error("Failed to open engine file: " + engine_path);
        // }
        // engine_file.seekg(0, std::ios::end);
        // size_t file_size = engine_file.tellg();
        // engine_file.seekg(0, std::ios::beg);
        // std::vector<char> engine_data(file_size);
        // engine_file.read(engine_data.data(), file_size);
        // engine_file.close();
        // engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
        //     runtime_->deserializeCudaEngine(engine_data.data(), file_size),
        //     [](nvinfer1::ICudaEngine* engine) { engine->destroy(); });

        // // Create execution context and allocate input/output buffers
        // context_ = engine_->createExecutionContext();
        // int num_bindings = engine_->getNbBindings();
        // buffers_.resize(num_bindings);
        // for (int i = 0; i < num_bindings; ++i)
        // {
        //     nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        //     size_t size = engine_->getBindingSize(i);
        //     cudaMalloc(&buffers_[i], size);
        //     if (engine_->bindingIsInput(i))
        //     {
        //         input_shapes_.emplace_back(dims.d[2], dims.d[1]);
        //         input_width_ = dims.d[2];
        //         input_height_ = dims.d[1];
        //         channels_ = dims.d[0];
        //     }
        // }
    }

    // Destructor
    ~YoloV8TRT()
    {
        for (void* buffer : buffers_)
        {
            cudaFree(buffer);
        }
        context_->destroy();
        runtime_->destroy();
    }

    void infer(const cv::Mat& image) override
    {
        // Preprocess the input image
        std::vector<float> input_data = preprocess_image(image);

        // // Copy input data to the GPU buffer
        // cudaMemcpy(buffers_[0], input_data.data(), input_data.size() * sizeof(float), cudaMemcpyHostToDevice);

        // // Run inference
        // context_->executeV2(buffers_.data());

        // // Get output tensor data
        // std::vector<float> output0_data(output_shapes_[0].total());
        // cudaMemcpy(output0_data.data(), buffers_[1], output0_data.size() * sizeof(float), cudaMemcpyDeviceToHost);

        // Process output data and draw predictions
        // ...

        // Release CUDA resources
        cudaDeviceSynchronize();
    }
};
