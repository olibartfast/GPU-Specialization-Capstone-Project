#include <onnxruntime_cxx_api.h>
#include "YoloV8.hpp"

struct Mask{
    std::vector<size_t> segShapes;
    std::vector<float> raw_mask_protos;
    std::vector<std::vector<float>> raw_mask_proposals;
};

struct Detection{

};

class YoloV8ONNX : public YoloV8{
private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::vector<std::string> input_names_; // Store input layer names
    std::vector<std::string> output_names_; // Store output layer names
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<std::vector<float>> input_tensors_;

    int input_width_{-1};
    int input_height_{-1};
    int channels_{-1};


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
        input_width_ = static_cast<int>(input_shapes_[0][3]);
        input_height_ = static_cast<int>(input_shapes_[0][2]);
        channels_ = static_cast<int>(input_shapes_[0][1]);
        std::cout << "channels " << channels_ << std::endl;
        std::cout << "w " << input_width_ << std::endl;
        std::cout << "h " << input_height_ << std::endl;


        // print name/shape of outputs
        std::cout << "Output Node Name/Shape (" << output_names_.size() << "):" << std::endl;
        for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
            output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
            auto output_shapes = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "\t" << output_names_.at(i) << " : " << print_shape(output_shapes) << std::endl;
            output_shapes_.emplace_back(output_shapes);
        }
       
    }

   ~YoloV8ONNX() {}

    std::vector<float> preprocess_image(const cv::Mat& image) {
        int target_width, target_height, offset_x, offset_y;
        float resize_ratio_width = static_cast<float>(input_width_) / static_cast<float>(image.cols);
        float resize_ratio_height = static_cast<float>(input_height_) / static_cast<float>(image.rows);

        if (resize_ratio_height > resize_ratio_width) {
            target_width = input_width_;
            target_height = resize_ratio_width * image.rows;
            offset_x = 0;
            offset_y = (input_height_ - target_height) / 2;
        } else {
            target_width = resize_ratio_height * image.cols;
            target_height = input_height_;
            offset_x = (input_width_ - target_width) / 2;
            offset_y = 0;
        }

        cv::Mat resized_image(target_height, target_width, CV_8UC3);
        cv::resize(image, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat output_image(input_width_, input_height_, CV_8UC3, cv::Scalar(128, 128, 128));
        resized_image.copyTo(output_image(cv::Rect(offset_x, offset_y, resized_image.cols, resized_image.rows)));

        // Split channels and permute to CHW format
        std::vector<cv::Mat> channels;
        cv::split(output_image, channels);

        std::vector<float> input_data(input_width_ * input_height_ * channels_);
        for (int c = 0; c < channels_; ++c) {
            cv::Mat channel = channels[c].reshape(1, input_width_ * input_height_);
            float* channel_data = reinterpret_cast<float*>(channel.data);
            std::copy(channel_data, channel_data + input_width_ * input_height_, input_data.begin() + c * input_width_ * input_height_);
        }

        return input_data;
    }

    void infer(const cv::Mat& image) override {
        // std::vector<Detection> detections;
        // Mask segmask;
        input_tensors_.emplace_back(preprocess_image(image));

        // TODO vector of blobs for batch_size > 1, mb blobFromImages
        std::vector<Ort::Value> in_ort_tensors;


        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

        for(size_t i = 0; i < session_.GetInputCount(); i++)
        {
            in_ort_tensors.push_back(Ort::Value::CreateTensor<float>(
                memoryInfo, input_tensors_[i].data(), input_tensors_[i].size(),
                input_shapes_[i].data(), input_shapes_[i].size()
                ));
        }

        // double-check the dimensions of the input tensor
        assert(in_ort_tensors[0].IsTensor() && in_ort_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shapes_[0]);
        std::cout << "\ninput_tensor shape: " << print_shape(in_ort_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;


        // pass data through model
        std::vector<const char*> input_names_char(input_names_.size(), nullptr);
        std::transform(std::begin(input_names_), std::end(input_names_), std::begin(input_names_char),
                        [&](const std::string& str) { return str.c_str(); });

        std::vector<const char*> output_names_char(output_names_.size(), nullptr);
        std::transform(std::begin(output_names_), std::end(output_names_), std::begin(output_names_char),
                        [&](const std::string& str) { return str.c_str(); });        
                
        std::vector<Ort::Value> output_ort_tensors = 
            session_.Run(Ort::RunOptions{nullptr}, input_names_char.data(), in_ort_tensors.data(), in_ort_tensors.size(),
             output_names_char.data(), output_names_.size());

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_ort_tensors.size() == output_names_.size() && output_ort_tensors[0].IsTensor());
            

        std::vector<float> output0(output_ort_tensors[0].GetTensorData<float>(),output_ort_tensors[0].GetTensorData<float>() + output_ort_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount());
        std::vector<float> output1(output_ort_tensors[1].GetTensorData<float>(), output_ort_tensors[1].GetTensorData<float>() + output_ort_tensors[1].GetTensorTypeAndShapeInfo().GetElementCount());  

        auto shapes0 = output_ort_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	    auto shapes1 = output_ort_tensors[1].GetTensorTypeAndShapeInfo().GetShape();    
        const auto offset = 4;
        const auto numClasses = shapes0[1] - offset -  shapes1[1];
        std::vector<std::vector<float>> output(shapes0[1], std::vector<float>(shapes0[2]));


    }
    // Add other necessary members and methods for ONNX Runtime inference
};
