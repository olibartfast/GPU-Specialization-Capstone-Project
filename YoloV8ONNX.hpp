#include <onnxruntime_cxx_api.h>
#include "YoloV8.hpp"

class YoloV8ONNX {
private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::vector<std::string> input_names_; // Store input layer names
    std::vector<std::string> output_names_; // Store output layer names
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;


public:

    // pretty prints a shape dimension vector
    std::string print_shape(const std::vector<std::int64_t>& v) 
    {
        std::stringstream ss("");
        for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
        ss << v[v.size() - 1];
        return ss.str();
    }


    YoloV8ONNX(const std::string& model_path) 
    {
        env_ =  Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YoloV8ONNX"); 
        // Initialize the ONNX Runtime environment
        Ort::SessionOptions session_options;
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_ = Ort::Session(env_, model_path.c_str(), session_options); // Load the model here

        Ort::AllocatorWithDefaultOptions allocator;
        std::cout << "Input Node Name/Shape (" << input_names_.size() << "):" << std::endl;
        for (std::size_t i = 0; i < session_.GetInputCount(); i++) 
        {
            input_names_.emplace_back(session_.GetInputNameAllocated(i, allocator).get());
            const auto input_shapes = session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "\t" << input_names_.at(i) << " : " << print_shape(input_shapes) << std::endl;
            input_shapes_.emplace_back(input_shapes);
        }


        // print name/shape of outputs
        std::cout << "Output Node Name/Shape (" << output_names_.size() << "):" << std::endl;
        for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
            output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
            auto output_shapes = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "\t" << output_names_.at(i) << " : " << print_shape(output_shapes) << std::endl;
            output_shapes_.emplace_back(output_shapes);
        }
       
    }


    void infer(const cv::Mat& image) {
        // Perform inference using the loaded ONNX model and ONNX Runtime API
        // Implement the logic to preprocess the image and feed it to the model
        // ...


    }
    // Add other necessary members and methods for ONNX Runtime inference
};
