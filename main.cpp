#include <iostream>
#include "cxxopts.hpp"
#include "common.hpp"

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
    while(cap.read(frame))
    {
        const auto detections = yolo->infer(frame);
        cv::Mat frame_with_mask = frame.clone();
        for (int i = 0; i < detections.size(); ++i) 
        {
            cv::rectangle(frame, detections[i].bbox, cv::Scalar(255, 0, 0));
            frame_with_mask(detections[i].bbox).setTo(cv::Scalar(255, 0, 0), detections[i].boxMask);
            
            // Write class label and score on the frame
            std::string label = classes[detections[i].label_id];
            float score = detections[i].score;
            std::string text = label + ": " + std::to_string(score);
            int baseline = 0;
            cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point textOrg(detections[i].bbox.x, detections[i].bbox.y - textSize.height);
            cv::putText(frame, text, textOrg, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
        }
        cv::addWeighted(frame, 0.5, frame_with_mask, 0.5, 0, frame);
        cv::imshow("", frame);
        cv::waitKey(1);
//#ifdef WRITE_FRAME
        outputVideo.write(frame);
//#endif        

    }


    return 0;
}
