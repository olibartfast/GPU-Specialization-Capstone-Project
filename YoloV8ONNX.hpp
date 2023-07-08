#include <onnxruntime_cxx_api.h>  // for ONNX Runtime C++ API
#include <onnxruntime_c_api.h>    // for CUDA execution provider (if using CUDA)
#include "YoloV8.hpp"


class YoloV8ONNX : public YoloV8
{
private:
    Ort::Env env_;
    Ort::Session session_{nullptr};
    std::vector<std::string> input_names_;  // Store input layer names
    std::vector<std::string> output_names_; // Store output layer names
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<std::vector<int64_t>> output_shapes_;

    int input_width_{-1};
    int input_height_{-1};
    int channels_{-1};

public:
    cv::Rect getSegPadSize(const size_t inputW,
    const size_t inputH,
    const cv::Size& inputSize
        )
    {
        std::vector<int> padSize;
        float w, h, x, y;
        float r_w = inputW / (inputSize.width * 1.0);
        float r_h = inputH / (inputSize.height * 1.0);
        if (r_h > r_w)
        {
            w = inputW;
            h = r_w * inputSize.height;
            x = 0;
            y = (inputH - h) / 2;
        }
        else
        {
            w = r_h * inputSize.width;
            h = inputH;
            x = (inputW - w) / 2;
            y = 0;
        }
        return cv::Rect(x, y, w,h);
    }

    cv::Rect get_rect(const cv::Size& imgSz, const std::vector<float>& bbox)
    {
        float r_w = input_width_ / static_cast<float>(imgSz.width);
        float r_h = input_height_ / static_cast<float>(imgSz.height);
        
        int l, r, t, b;
        if (r_h > r_w) {
            l = bbox[0] - bbox[2] / 2.f;
            r = bbox[0] + bbox[2] / 2.f;
            t = bbox[1] - bbox[3] / 2.f - (input_height_ - r_w * imgSz.height) / 2;
            b = bbox[1] + bbox[3] / 2.f - (input_height_ - r_w * imgSz.height) / 2;
            l /= r_w;
            r /= r_w;
            t /= r_w;
            b /= r_w;
        }
        else {
            l = bbox[0] - bbox[2] / 2.f - (input_width_ - r_h * imgSz.width) / 2;
            r = bbox[0] + bbox[2] / 2.f - (input_width_ - r_h * imgSz.width) / 2;
            t = bbox[1] - bbox[3] / 2.f;
            b = bbox[1] + bbox[3] / 2.f;
            l /= r_h;
            r /= r_h;
            t /= r_h;
            b /= r_h;
        }
        
        // Clamp the coordinates within the image bounds
        l = std::max(0, std::min(l, imgSz.width - 1));
        r = std::max(0, std::min(r, imgSz.width - 1));
        t = std::max(0, std::min(t, imgSz.height - 1));
        b = std::max(0, std::min(b, imgSz.height - 1));

        return cv::Rect(l, t, r - l, b - t);
    }

    // pretty prints a shape dimension vector
    std::string print_shape(const std::vector<std::int64_t> &v)
    {
        std::stringstream ss("");
        for (std::size_t i = 0; i < v.size() - 1; i++)
            ss << v[i] << "x";
        ss << v[v.size() - 1];
        return ss.str();
    }

    YoloV8ONNX(const std::string &model_path, bool use_gpu = false)
    {

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YoloV8ONNX");

        Ort::SessionOptions session_options;

        if(use_gpu)
        {
            // Check if CUDA GPU is available
            int device_id = 0;
            OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);
            if (status == nullptr) {
                // CUDA GPU is available, use it
                std::cout << "Using CUDA GPU" << std::endl;
            } else {
                // CUDA GPU is not available, fall back to CPU
                std::cout << "CUDA GPU not available, falling back to CPU" << std::endl;
                Ort::GetApi().ReleaseStatus(status);
                session_options = Ort::SessionOptions();
            }
        }

        session_ = Ort::Session(env, model_path.c_str(), session_options);

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
        for (std::size_t i = 0; i < session_.GetOutputCount(); i++)
        {
            output_names_.emplace_back(session_.GetOutputNameAllocated(i, allocator).get());
            auto output_shapes = session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
            std::cout << "\t" << output_names_.at(i) << " : " << print_shape(output_shapes) << std::endl;
            output_shapes_.emplace_back(output_shapes);
        }
    }

    std::vector<float> preprocess_image(const cv::Mat &image)
    {
        cv::Mat blob;
        cv::cvtColor(image, blob, cv::COLOR_BGR2RGB);
        int target_width, target_height, offset_x, offset_y;
        float resize_ratio_width = static_cast<float>(input_width_) / static_cast<float>(image.cols);
        float resize_ratio_height = static_cast<float>(input_height_) / static_cast<float>(image.rows);

        if (resize_ratio_height > resize_ratio_width)
        {
            target_width = input_width_;
            target_height = resize_ratio_width * image.rows;
            offset_x = 0;
            offset_y = (input_height_ - target_height) / 2;
        }
        else
        {
            target_width = resize_ratio_height * image.cols;
            target_height = input_height_;
            offset_x = (input_width_ - target_width) / 2;
            offset_y = 0;
        }

        cv::Mat resized_image(target_height, target_width, CV_8UC3);
        cv::resize(blob, resized_image, resized_image.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat output_image(input_width_, input_height_, CV_8UC3, cv::Scalar(128, 128, 128));
        resized_image.copyTo(output_image(cv::Rect(offset_x, offset_y, resized_image.cols, resized_image.rows)));   
        output_image.convertTo(output_image, CV_32FC3, 1.f / 255.f);        

        size_t img_byte_size = output_image.total() * output_image.elemSize();  // Allocate a buffer to hold all image elements.
        std::vector<float> input_data = std::vector<float>(input_width_ * input_height_ * channels_);
        std::memcpy(input_data.data(), output_image.data, img_byte_size);

        std::vector<cv::Mat> chw;
        for (size_t i = 0; i < channels_; ++i)
        {
            chw.emplace_back(cv::Mat(cv::Size(input_width_, input_height_), CV_32FC1, &(input_data[i * input_width_ * input_height_])));
        }
        cv::split(output_image, chw);

        return input_data;
    }

  void infer(const cv::Mat& image) override {
    
    std::vector<std::vector<float>> input_tensors(session_.GetInputCount());
    std::vector<Ort::Value> in_ort_tensors;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    for (size_t i = 0; i < session_.GetInputCount(); ++i) {
        input_tensors[i] = preprocess_image(image);
        in_ort_tensors.emplace_back(Ort::Value::CreateTensor<float>(
            memory_info,
            input_tensors[i].data(),
            input_tensors[i].size(),
            input_shapes_[i].data(),
            input_shapes_[i].size()
        ));
    }

    // Run inference
    std::vector<const char*> input_names_char(input_names_.size());
    std::transform(input_names_.begin(), input_names_.end(), input_names_char.begin(),
        [](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names_.size());
    std::transform(output_names_.begin(), output_names_.end(), output_names_char.begin(),
        [](const std::string& str) { return str.c_str(); });

    std::vector<Ort::Value> output_ort_tensors = session_.Run(
        Ort::RunOptions{ nullptr },
        input_names_char.data(),
        in_ort_tensors.data(),
        in_ort_tensors.size(),
        output_names_char.data(),
        output_names_.size()
    );

    // Process output tensors
    assert(output_ort_tensors.size() == output_names_.size());

    const auto& output0 = output_ort_tensors[0].GetTensorData<float>();
    const auto& output1 = output_ort_tensors[1].GetTensorData<float>();

    const auto& shape0 = output_ort_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    const auto& shape1 = output_ort_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

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
            std::cout << label<<std::endl;
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
        maskProposals.push_back( cv::Mat(raw_mask_proposals[i]).t() );
    }

		

    cv::Mat frame = image.clone();
    cv::Mat frame_with_mask = image.clone();
	for (int i = 0; i < detections.size(); ++i) {

        int sc, sh, sw;
        std::tie(sc, sh, sw) = std::make_tuple(static_cast<int>(shape1[1]), static_cast<int>(shape1[2]), static_cast<int>(shape1[3]));
        cv::Mat protos = cv::Mat(std::vector<float>(output1, output1 + sc*sh*sw)).reshape(0, { sc, sw*sh}); 
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
}

};
