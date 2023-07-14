#include <iostream>
#include "cxxopts.hpp"
#include "common.hpp"
#include "labels.hpp"
#include <chrono>

// Include the appropriate header based on the selected framework
#ifdef USE_ONNX_RUNTIME
#include "YoloV8ONNX.hpp"
#elif USE_LIBTORCH
#include "YoloV8Libtorch.hpp"
#elif USE_TENSORRT
#include "YoloV8TRT.hpp"
#endif

std::unique_ptr<YoloV8> createYoloV8(const std::string& weights, const bool is_gpu) {
#ifdef USE_ONNX_RUNTIME
    return std::make_unique<YoloV8ONNX>(weights, is_gpu);
#elif USE_LIBTORCH
    return std::make_unique<YoloV8Libtorch>(weights, is_gpu);
#elif USE_TENSORRT
    return std::make_unique<YoloV8TRT>(weights);
#else
    return nullptr;
#endif
}

void processFrame(YoloV8* yolo, cv::Mat& frame, const std::vector<cv::Scalar>& class_colors, const std::vector<std::string>& classes, bool enable_imshow) {
    const auto [detections, segMask] = yolo->infer(frame);
    cv::Mat frame_with_mask = frame.clone();
    const auto maskProposals = segMask.maskProposals;
    const auto protos = segMask.protos;
    const auto roi = segMask.maskRoi;
    cv::Mat masks;
    std::vector<cv::Mat> maskChannels;
    if (!detections.empty()) {
        masks = cv::Mat((maskProposals * protos).t()).reshape(detections.size(), {160, 160});
        cv::split(masks, maskChannels);
    }

    for (int i = 0; i < detections.size(); ++i) {
        cv::Mat mask;

        // Sigmoid
        cv::exp(-maskChannels[i], mask);
        mask = 1.0 / (1.0 + mask); // 160*160

        mask = mask(roi);
        const auto class_id = detections[i].label_id;
        cv::resize(mask, mask, cv::Size(frame.cols, frame.rows), cv::INTER_NEAREST);
        const float mask_thresh = 0.5f;
        mask = mask(detections[i].bbox) > mask_thresh;
        cv::rectangle(frame, detections[i].bbox, class_colors[class_id], 2);
        frame_with_mask(detections[i].bbox).setTo(class_colors[class_id], mask);

        // Write class label and score on the frame
        std::string label = classes[detections[i].label_id];
        float score = detections[i].score;
        std::string text = label + ": " + std::to_string(score);
        int baseline = 0;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::Point textOrg(detections[i].bbox.x, detections[i].bbox.y - textSize.height);
        cv::putText(frame, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, class_colors[class_id], 2, cv::LINE_AA);
    }
    cv::addWeighted(frame, 0.5, frame_with_mask, 0.5, 0, frame);

    // Display frame if enabled
    if (enable_imshow) {
        cv::imshow("", frame);
        cv::waitKey(1);
    }
}

void processVideo(const std::string& video, YoloV8* yolo,
 const std::vector<cv::Scalar>& class_colors, const std::vector<std::string>& classes, 
 const std::string& backend,
 bool is_gpu,
 bool enable_imshow, bool enable_videowrite) {
    cv::VideoCapture cap(video);
    cv::Mat frame;

    // Write video to file if enabled
    cv::VideoWriter outputVideo;
    if (enable_videowrite) {
        cap.read(frame);
        outputVideo.open("output_video.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 25, frame.size());

        if (!outputVideo.isOpened()) {
            std::cerr << "Failed to open the video writer." << std::endl;
            return;
        }

        // Write frame to video
        outputVideo.write(frame);
    }

    // Variables for FPS calculation
    int frame_count = 0;
    double total_time = 0.0;
    double average_fps = 0.0;

    while (cap.read(frame)) {
        auto start_time = std::chrono::steady_clock::now();

        // Process the frame
        processFrame(yolo, frame, class_colors, classes, enable_imshow);

        // Calculate inference time and FPS
        auto end_time = std::chrono::steady_clock::now();
        double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
        double fps = 1.0 / frame_time;

        // Accumulate frame time and count
        total_time += frame_time;
        frame_count++;

        // Compute average FPS
        average_fps = frame_count / total_time;

        // Print FPS on the frame
        std::string fps_text = "fps: " + std::to_string(fps);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);

        // Print backend, model name, and GPU flag on the frame
        std::string backend_text = "Backend: " + backend;
        std::string gpu_text = "GPU: " + std::string(is_gpu ? "true" : "false");
        cv::putText(frame, backend_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, gpu_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);



        // Display frame if enabled
        if (enable_imshow) {
            cv::imshow("", frame);
            cv::waitKey(1);
        }

        // Write frame to video if enabled
        if (enable_videowrite)
            outputVideo.write(frame);
    }

    // Print the average FPS
    std::cout << "Average FPS: " << average_fps << std::endl;
}

int main(int argc, char* argv[]) {
    cxxopts::Options options("UnderwaterTrashInstanceSegmentation", "Description of your program");

    // Add command line options
    options.add_options()
        ("w,weights", "Path to weights", cxxopts::value<std::string>())
        ("v,video", "Path to video source", cxxopts::value<std::string>())
        ("gpu", "Is GPU wanted", cxxopts::value<bool>()->default_value("false"))
        ("videowrite", "Enable video writing", cxxopts::value<bool>()->default_value("false"))
        ("imshow", "Enable frame display", cxxopts::value<bool>()->default_value("true"))
        ("h,help", "Print help");

    // Parse command line arguments
    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (!result.count("weights")) {
        std::cerr << "Weights not specified. Use the --weights option." << std::endl;
        return 1;
    }

    if (!result.count("video")) {
        std::cerr << "video input source not specified. Use the --video option." << std::endl;
        return 1;
    }

    if (!result.count("gpu")) {
        std::cerr << "gpu input not specified. Use the --gpu option." << std::endl;
        return 1;
    }

    std::string weights = result["weights"].as<std::string>();
    std::string video = result["video"].as<std::string>();
    bool is_gpu = result["gpu"].as<bool>();
    bool enable_videowrite = result["videowrite"].as<bool>();
    bool enable_imshow = result["imshow"].as<bool>();

  

    // Extract the extension from the weight file path
    std::string backend;
    std::string weight_file_extension = result["weights"].as<std::string>();
    size_t extension_pos = weight_file_extension.rfind('.');
    if (extension_pos != std::string::npos)
    {
        std::string extension = weight_file_extension.substr(extension_pos + 1);
        std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

        if (extension == "onnx")
        {
            // Use ONNX Runtime as the backend
            backend = "ONNX_RUNTIME";
        }
        else if (extension == "pt" || extension == "pth" || extension == "torchscript")
        {
            // Use LibTorch as the backend
            backend = "LIBTORCH";
        }
        else if (extension == "trt" || extension == "engine" || extension == "plan")
        {
            // Use TensorRT as the backend
            backend = "TENSORRT";
        }
        else
        {
            std::cerr << "Invalid weight file extension. Supported extensions: .onnx, .pt, .pth, .trt" << std::endl;
            std::exit(1);
        }
    }
    else
    {
        std::cerr << "Invalid weight file path" << std::endl;
        std::exit(1);
    }

    std::unique_ptr<YoloV8> yolo = createYoloV8(weights, is_gpu);
    if (!yolo) {
        std::cerr << "Invalid framework specified. Supported frameworks are ONNX_RUNTIME, LIBTORCH, and TENSORRT." << std::endl;
        return 1;
    }

    processVideo(video, yolo.get(), class_colors, classes, backend, is_gpu, enable_imshow, enable_videowrite);

    return 0;
}
