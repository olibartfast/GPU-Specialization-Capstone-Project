#include <iostream>
#include "cxxopts.hpp"
#include "common.hpp"
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

int main(int argc, char* argv[]) {
    cxxopts::Options options("UnderwaterTrashInstanceSegmentation", "Description of your program");

    // Add command line options
    options.add_options()
        ("w,weights", "Path to weights", cxxopts::value<std::string>())
        ("v,video", "Path to video source", cxxopts::value<std::string>())
        ("gpu", "Is gpu wanted", cxxopts::value<bool>()->default_value("false"))
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

    std::vector<std::string> classes = {
        "rov",
        "plant",
        "animal_fish",
        "animal_starfish",
        "animal_shells",
        "animal_crab",
        "animal_eel",
        "animal_etc",
        "trash_etc",
        "trash_fabric",
        "trash_fishing_gear",
        "trash_metal",
        "trash_paper",
        "trash_plastic",
        "trash_rubber",
        "trash_wood"
    };    


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


    std::unique_ptr<YoloV8> yolo = createYoloV8(weights,is_gpu);
    if (!yolo) {
        std::cerr << "Invalid framework specified. Supported frameworks are ONNX_RUNTIME, LIBTORCH, and TENSORRT." << std::endl;
        return 1;
    }
    cv::VideoCapture cap(video);
    cv::Mat frame;

// #ifdef WRITE_FRAME
    cv::VideoWriter outputVideo;
    cv::Size S = cv::Size((int)cap.get(cv::CAP_PROP_FRAME_WIDTH),    // Acquire input size
        (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    outputVideo.open("processed.avi", codec, cap.get(cv::CAP_PROP_FPS), S, true);
    if (!outputVideo.isOpened()) {
        std::cout << "Could not open the output video for write: " << video << std::endl;
        return -1;
    }
//#endif

    // Variables for FPS calculation
    int frame_count = 0;
    double total_time = 0.0;
    double average_fps = 0.0;

    while(cap.read(frame))
    {
        auto start_time = std::chrono::steady_clock::now();
        const auto [detections, segMask] = yolo->infer(frame);
        cv::Mat frame_with_mask = frame.clone();
        const auto maskProposals = segMask.maskProposals;
        const auto protos = segMask.protos;
        const auto roi = segMask.maskRoi;
        for (int i = 0; i < detections.size(); ++i) 
        {
            cv::Mat masks = cv::Mat((maskProposals[i] * protos).t()).reshape(detections.size(), {160, 160 });
            std::vector<cv::Mat> maskChannels;
            cv::split(masks, maskChannels);
            cv::Mat mask;

            // Sigmoid
            cv::exp(-maskChannels[i], mask);
            mask = 1.0 / (1.0 + mask); // 160*160
        
            mask = mask(roi);
            cv::resize(mask, mask, cv::Size(frame.cols, frame.rows), cv::INTER_NEAREST);
            const float mask_thresh = 0.5f;
            mask = mask(detections[i].bbox) > mask_thresh;
            cv::rectangle(frame, detections[i].bbox, cv::Scalar(255, 0, 0));
            frame_with_mask(detections[i].bbox).setTo(cv::Scalar(255, 0, 0), mask);
            
            // Write class label and score on the frame
            std::string label = classes[detections[i].label_id];
            float score = detections[i].score;
            std::string text = label + ": " + std::to_string(score);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point textOrg(detections[i].bbox.x, detections[i].bbox.y - textSize.height);
            cv::putText(frame, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        }
        cv::addWeighted(frame, 0.5, frame_with_mask, 0.5, 0, frame);

        // Calculate inference time and FPS
        // Measure the time taken for inference
        auto end_time = std::chrono::steady_clock::now();
        double frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;

        // Compute FPS
        double fps = 1.0 / frame_time;

        // Accumulate frame time and count
        total_time += frame_time;
        frame_count++;

        // Compute average FPS
        average_fps = frame_count / total_time;

        // Print FPS on the frame
        std::string fps_text = "FPS: " + std::to_string(fps);
        cv::putText(frame, fps_text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

        // Print backend, model name, and GPU flag on the frame
        std::string backend_text = "Backend: " + backend;
        std::string gpu_text = "GPU: " + std::string(is_gpu ? "true" : "false");
        cv::putText(frame, backend_text, cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        cv::putText(frame, gpu_text, cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

        cv::imshow("", frame);
        cv::waitKey(1);
        //#ifdef WRITE_FRAME
        outputVideo.write(frame);
        //#endif
    }

    // Print the average FPS
    std::cout << "Average FPS: " << average_fps << std::endl;



    return 0;
}
