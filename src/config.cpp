#include "../include/config.h"
#include <iostream>

Config loadConfig(const std::string &filename) {
    YAML::Node config = YAML::LoadFile(filename);
    Config cfg;

    // Tracker
    cfg.reprojection_window = config["Tracker"]["reprojection_window"].as<int>();
    
    // ORB parameters
    cfg.num_features = config["ORB"]["num_features"].as<int>();
    std::cout << cfg.reprojection_window << "\n\n\n";
    
    // PnP RANSAC parameters
    cfg.ransac_iterations = config["PnP"]["ransac_iterations"].as<int>();
    cfg.reprojection_error = config["PnP"]["reprojection_error"].as<double>();
    
    auto K_yaml = config["Camera"]["K"];
    cfg.K = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cfg.K.at<double>(i, j) = K_yaml[i][j].as<double>();

    auto distortion_yaml = config["Camera"]["distortion"];
    for (int i = 0; i < 5; i++) {
        cfg.distortion.push_back(distortion_yaml[i].as<float>());
    }

    return cfg;
}
