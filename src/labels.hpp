  #pragma once
  #include "common.hpp"
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

    // Define colors for each class label
    std::vector<cv::Scalar> class_colors = {
        cv::Scalar(255, 0, 0),     // Class 0: rov (Blue color)
        cv::Scalar(0, 255, 0),     // Class 1: plant (Green color)
        cv::Scalar(0, 0, 255),     // Class 2: animal_fish (Red color)
        cv::Scalar(255, 255, 0),   // Class 3: animal_starfish (Cyan color)
        cv::Scalar(255, 0, 255),   // Class 4: animal_shells (Magenta color)
        cv::Scalar(0, 255, 255),   // Class 5: animal_crab (Yellow color)
        cv::Scalar(255, 128, 0),   // Class 6: animal_eel (Orange color)
        cv::Scalar(128, 255, 0),   // Class 7: animal_etc (Light Green color)
        cv::Scalar(0, 128, 255),   // Class 8: trash_etc (Light Blue color)
        cv::Scalar(0, 0, 0),       // Class 9: trash_fabric (Black color)
        cv::Scalar(255, 255, 255), // Class 10: trash_fishing_gear (White color)
        cv::Scalar(128, 128, 128), // Class 11: trash_metal (Gray color)
        cv::Scalar(0, 0, 128),     // Class 12: trash_paper (Dark Blue color)
        cv::Scalar(128, 0, 0),     // Class 13: trash_plastic (Dark Red color)
        cv::Scalar(0, 128, 0),     // Class 14: trash_rubber (Dark Green color)
        cv::Scalar(128, 0, 128)    // Class 15: trash_wood (Purple color)
    };