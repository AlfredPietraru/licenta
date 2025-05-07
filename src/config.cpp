#include "../include/config.h"
#include <iostream>

Config loadConfig(const std::string &filename) {
    YAML::Node config = YAML::LoadFile(filename);
    Config cfg;

    auto K_yaml = config["Camera"]["K"];
    cfg.K = cv::Mat(3, 3, CV_64F);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            cfg.K.at<double>(i, j) = K_yaml[i][j].as<double>();

    auto distortion_yaml = config["Camera"]["distortion"];
    for (int i = 0; i < 5; i++) {
        cfg.distortion.push_back(distortion_yaml[i].as<float>());
    }
    
    cfg.orb_matcher.orb_descriptor_value = config["Matcher"]["orb_descriptor_value"].as<int>();
    cfg.orb_matcher.window = config["Matcher"]["window"].as<int>();
    cfg.orb_matcher.minim_points_found = config["Matcher"]["minim_points_found"].as<int>();
    cfg.orb_matcher.ratio_key_frame_match = config["Matcher"]["ratio_key_frame_match"].as<double>();
    cfg.orb_matcher.ratio_track_local_map = config["Matcher"]["ratio_track_local_map"].as<double>();

    cfg.pnp_ransac_config.reprojection_window = config["PnP"]["reprojection_window"].as<int>();
    cfg.pnp_ransac_config.ransac_iterations = config["PnP"]["ransac_iterations"].as<int>();
    cfg.pnp_ransac_config.confidence = config["PnP"]["confidence"].as<double>();
    

    return cfg;
}
