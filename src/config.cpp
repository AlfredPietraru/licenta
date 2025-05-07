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
    Orb_Matcher orb_matcher_cfg;
    orb_matcher_cfg.orb_descriptor_value = config["Matcher"]["orb_descriptor_value"].as<int>();
    orb_matcher_cfg.des_dist_high = config["Matcher"]["des_dist_high"].as<int>();
    orb_matcher_cfg.des_dist_low = config["Matcher"]["des_dist_low"].as<int>();
    orb_matcher_cfg.ratio_track_local_map = config["Matcher"]["ratio_track_local_map"].as<double>();
    orb_matcher_cfg.match_reference_frame_orb_descriptor_ratio = config["Matcher"]["match_reference_frame_orb_descriptor_ratio"].as<double>();
    cfg.orb_matcher = orb_matcher_cfg;
    return cfg;
}
