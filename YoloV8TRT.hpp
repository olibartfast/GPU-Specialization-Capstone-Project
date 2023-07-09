#pragma once
#include "common.hpp"
#include <NvInfer.h>  // for TensorRT API
#include <cuda_runtime_api.h>  // for CUDA runtime API
#include <fstream>

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Implement logging behavior here, e.g., print the log message
        if (severity != Severity::kINFO)
        {
            std::cout << "TensorRT Logger: " << msg << std::endl;
        }
    }
};

class YoloV8TRT : public YoloV8
{
private:
    std::shared_ptr<nvinfer1::ICudaEngine> engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    std::vector<void*> buffers_;
    std::vector<cv::Size> input_shapes_;

    nvinfer1::IRuntime* runtime_{nullptr};

public:
    // Constructor
    YoloV8TRT(const std::string& engine_path)
    {
        Logger logger;
        runtime_ = nvinfer1::createInferRuntime(logger);
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file)
        {
            throw std::runtime_error("Failed to open engine file: " + engine_path);
        }
        engine_file.seekg(0, std::ios::end);
        size_t file_size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        std::vector<char> engine_data(file_size);
        engine_file.read(engine_data.data(), file_size);
        engine_file.close();
        engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
            runtime_->deserializeCudaEngine(engine_data.data(), file_size),
            [](nvinfer1::ICudaEngine* engine) { engine->destroy(); });

        // Create execution context and allocate input/output buffers
        context_ = engine_->createExecutionContext();
        int num_bindings = engine_.get()->getNbBindings();
        buffers_.resize(num_bindings);
        for (int i = 0; i < num_bindings; ++i)
        {
            nvinfer1::Dims dims = engine_.get()->getBindingDimensions(i);
            if (engine_.get()->bindingIsInput(i))
            {
                const auto input_shape = std::vector{dims.d[0], dims.d[1], dims.d[2], dims.d[3]};
                input_width_ = dims.d[3];
                input_height_ = dims.d[2];
                channels_ = dims.d[1];
            }
        }
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
