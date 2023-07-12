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

        const auto offset = 4;
        const auto num_classes = shape0[1] - offset - shape1[1];
        std::vector<std::vector<float>> output0_matrix(shape0[1], std::vector<float>(shape0[2]));

        // Construct output matrix
        for (int i = 0; i < shape0[1]; ++i) {
            for (int j = 0; j < shape0[2]; ++j) {
                output0_matrix[i][j] = output0[i * shape0[2] + j];
            }
        }

        std::vector<std::vector<float>> transposed_output0(shape0[2], std::vector<float>(shape0[1]));

        // Transpose output matrix
        for (int i = 0; i < shape0[1]; ++i) {
            for (int j = 0; j < shape0[2]; ++j) {
                transposed_output0[j][i] = output0_matrix[i][j];
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
        for (int i = 0; i < shape0[2]; ++i) {
            const auto& row = transposed_output0[i];
            const float* bboxesPtr = row.data();
            const float* scoresPtr = bboxesPtr + 4;
            auto maxSPtr = std::max_element(scoresPtr, scoresPtr + num_classes);
            float score = *maxSPtr;
            if (score > conf_threshold) {
                boxes.emplace_back(get_rect(frame_size, std::vector<float>(bboxesPtr, bboxesPtr + 4)));
                int label = maxSPtr - scoresPtr;
                std::cout << label << std::endl;
                confs.emplace_back(score);
                classIds.emplace_back(label);
                picked_proposals.emplace_back(std::vector<float>(scoresPtr + num_classes, scoresPtr + num_classes + shape1[1]));
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
            maskProposals.push_back(cv::Mat(raw_mask_proposals[i]).t());
        }

        cv::Mat frame = image.clone();
        cv::Mat frame_with_mask = image.clone();
        for (int i = 0; i < detections.size(); ++i) {

            int sc, sh, sw;
            std::tie(sc, sh, sw) = std::make_tuple(static_cast<int>(shape1[1]), static_cast<int>(shape1[2]), static_cast<int>(shape1[3]));
            cv::Mat protos = cv::Mat(std::vector<float>(output1, output1 + sc * sh * sw)).reshape(0, { sc, sw * sh });
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
            cv::rectangle(frame, temp_rect, cv::Scalar(255, 0, 0));
            const float mask_thresh = 0.5f;
            mask = mask(temp_rect) > mask_thresh;
            frame_with_mask(temp_rect).setTo(cv::Scalar(255, 0, 0), mask);
        }

        cv::addWeighted(frame, 0.5, frame_with_mask, 0.5, 0, frame);

        cv::imshow("", frame);
        cv::waitKey(1);
    }
};
