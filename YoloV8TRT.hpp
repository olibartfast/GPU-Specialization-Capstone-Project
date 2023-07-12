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
    std::vector<nvinfer1::Dims> output_dims_; // and one output
    std::vector<std::vector<float>> h_outputs_;
    nvinfer1::IRuntime* runtime_{nullptr};

public:
    // Create the TensorRT runtime and deserialize the engine
    std::shared_ptr<nvinfer1::ICudaEngine> createRuntimeAndDeserializeEngine(const std::string& engine_path, Logger& logger, nvinfer1::IRuntime*& runtime)
    {
        // Create TensorRT runtime
        runtime = nvinfer1::createInferRuntime(logger);

        // Load engine file
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

        // Deserialize engine
        std::shared_ptr<nvinfer1::ICudaEngine> engine(
            runtime->deserializeCudaEngine(engine_data.data(), file_size),
            [](nvinfer1::ICudaEngine* engine) { engine->destroy(); });

        return engine;
    }

    // Create execution context and allocate input/output buffers
    void createContextAndAllocateBuffers(nvinfer1::ICudaEngine* engine, nvinfer1::IExecutionContext*& context, std::vector<void*>& buffers, std::vector<nvinfer1::Dims>& output_dims, std::vector<std::vector<float>>& h_outputs)
    {
        context = engine->createExecutionContext();
        buffers.resize(engine->getNbBindings());
        for (int i = 0; i < engine->getNbBindings(); ++i)
        {
            nvinfer1::Dims dims = engine->getBindingDimensions(i);
            auto binding_size = getSizeByDim(engine->getBindingDimensions(i)) * sizeof(float);
            cudaMalloc(&buffers[i], binding_size);
            if (engine->bindingIsInput(i))
            {
                const auto input_shape = std::vector{ dims.d[0], dims.d[1], dims.d[2], dims.d[3] };
                input_width_ = dims.d[3];
                input_height_ = dims.d[2];
                channels_ = dims.d[1];
            }
            else
            {
                output_dims.emplace_back(engine->getBindingDimensions(i));
                auto size = getSizeByDim(dims);
                h_outputs.emplace_back(std::vector<float>(size));
            }
        }
    }

    void initializeBuffers(const std::string& engine_path)
    {
        // Create logger
        Logger logger;

        // Create TensorRT runtime and deserialize engine
        engine_ = createRuntimeAndDeserializeEngine(engine_path, logger, runtime_);

        // Create execution context and allocate input/output buffers
        createContextAndAllocateBuffers(engine_.get(), context_, buffers_, output_dims_, h_outputs_);
    }

    // calculate size of tensor
    size_t getSizeByDim(const nvinfer1::Dims& dims)
    {
        size_t size = 1;
        for (size_t i = 0; i < dims.nbDims; ++i)
        {
            std::cout << dims.d[i] << std::endl;
            size *= dims.d[i];
        }
        return size;
    }

    YoloV8TRT(const std::string& engine_path)
    {
        initializeBuffers(engine_path);
    }

    // Destructor
    ~YoloV8TRT()
    {
        for (void* buffer : buffers_)
        {
            cudaFree(buffer);
        }
    }

    void infer(const cv::Mat& image) override
    {
        // Preprocess the input image
        std::vector<float> h_input_data = preprocess_image(image);
        cudaMemcpy(buffers_[0], h_input_data.data(), sizeof(float)*h_input_data.size(), cudaMemcpyHostToDevice);

        if(context_->enqueueV2(buffers_.data(), 0, nullptr))
            std::cout << "Forward success !" << std::endl;
         else
            std::cout << "Forward Error !" << std::endl;

        for (size_t i = 0; i < h_outputs_.size(); i++)
            cudaMemcpy(h_outputs_[i].data(), buffers_[i + 1], h_outputs_[i].size() * sizeof(float), cudaMemcpyDeviceToHost);

        const float* output_mask = h_outputs_[0].data();
        const float* output_boxes = h_outputs_[1].data();

        const int* shape_mask = reinterpret_cast<const int*>(output_dims_[0].d);
        const int* shape_boxes = reinterpret_cast<const int*>(output_dims_[1].d);


        const auto offset = 4;
        const auto num_classes = shape_boxes[1] - offset - shape_mask[1];
        std::vector<std::vector<float>> output_boxes_matrix(shape_boxes[1], std::vector<float>(shape_boxes[2]));

        // Construct output matrix
        for (int i = 0; i < shape_boxes[1]; ++i) {
            for (int j = 0; j < shape_boxes[2]; ++j) {
                output_boxes_matrix[i][j] = output_boxes[i * shape_boxes[2] + j];
            }
        }

        std::vector<std::vector<float>> transposed_output_boxes(shape_boxes[2], std::vector<float>(shape_boxes[1]));

        // Transpose output matrix
        for (int i = 0; i < shape_boxes[1]; ++i) {
            for (int j = 0; j < shape_boxes[2]; ++j) {
                transposed_output_boxes[j][i] = output_boxes_matrix[i][j];
            }
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        std::vector<int> classIds;
        const auto conf_threshold = 0.25f;
        const auto iou_threshold = 0.4f;
        cv::Size frame_size(image.cols, image.rows);
        std::vector<std::vector<float>> picked_proposals;

        // Get all the YOLO proposals
        for (int i = 0; i < shape_boxes[2]; ++i) {
            const auto& row = transposed_output_boxes[i];
            const float* bboxesPtr = row.data();
            const float* scoresPtr = bboxesPtr + 4;
            auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
            float score = *maxSPtr;
            if (score > conf_threshold) {
                boxes.emplace_back(get_rect(frame_size, std::vector<float>(bboxesPtr, bboxesPtr + 4)));
                int label = maxSPtr - scoresPtr;
                std::cout << label<<std::endl;
                confs.emplace_back(score);
                classIds.emplace_back(label);
                picked_proposals.emplace_back(std::vector<float>(scoresPtr + num_classes, scoresPtr + num_classes + shape_mask[1]));
            }
        }

        // Perform Non Maximum Suppression and draw predictions.
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confs, conf_threshold, iou_threshold, indices);
        std::vector<Detection> detections;
        std::vector<std::vector<float>> raw_mask_proposals;
        cv::Mat maskProposals;
        for (int i = 0; i < indices.size(); i++) 
        {
            Detection det;
            int idx = indices[i];
            det.label_id = classIds[idx];
            det.bbox = boxes[idx];
            det.score = confs[idx];
            detections.emplace_back(det);
            raw_mask_proposals.push_back(picked_proposals[idx]);
            maskProposals.push_back( cv::Mat(raw_mask_proposals[i]).t() );
        }

            

        cv::Mat frame = image.clone();
        cv::Mat frame_with_mask = image.clone();
        for (int i = 0; i < detections.size(); ++i) {

            int sc, sh, sw;
            std::tie(sc, sh, sw) = std::make_tuple(static_cast<int>(shape_mask[1]), static_cast<int>(shape_mask[2]), static_cast<int>(shape_mask[3]));
            cv::Mat protos = cv::Mat(std::vector<float>(output_mask, output_mask + sc*sh*sw)).reshape(0, { sc, sw*sh}); 
            cv::Mat masks = cv::Mat((maskProposals * protos).t()).reshape(detections.size(), { sw, sh });
            std::vector<cv::Mat> maskChannels;
            cv::split(masks, maskChannels);           
    
            cv::Mat mask;

            // Sigmoid
            cv::exp(-maskChannels[i], mask);
            mask = 1.0 / (1.0 + mask); // 160*160

            cv::Rect segPadRect = getSegPadSize(input_width_, input_height_, frame_size);
            cv::Rect roi(int((float)segPadRect.x / input_width_ * sw), int((float)segPadRect.y / input_height_ * sh), int(sw - segPadRect.x / 2), int(sh - segPadRect.y / 2));
            mask = mask(roi);
            cv::resize(mask, mask, image.size(), cv::INTER_NEAREST);

            cv::Rect temp_rect = detections[i].bbox;
            cv::rectangle(frame, temp_rect, cv::Scalar(255,0,0));
            const float mask_thresh = 0.5f;
            mask = mask(temp_rect) > mask_thresh;		
            frame_with_mask(temp_rect).setTo(cv::Scalar(255,0,0), mask);
        }    

        cv::addWeighted(frame, 0.5, frame_with_mask, 0.5, 0, frame); 

        cv::imshow("", frame);
        cv::waitKey(1);


        // Release CUDA resources
        cudaDeviceSynchronize();
    }
};
 