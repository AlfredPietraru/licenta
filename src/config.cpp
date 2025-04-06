#include "../include/config.h"
#include <iostream>


Orb_Matcher load_orb_matcher_config(const std::string &filename) {
    YAML::Node config = YAML::LoadFile(filename);
    Orb_Matcher cfg;
    cfg.orb_descriptor_value = config["Matcher"]["orb_descriptor_value"].as<int>();
    cfg.window = config["Matcher"]["window"].as<int>();
    cfg.minim_points_found = config["Matcher"]["minim_points_found"].as<int>();
    cfg.ratio_key_frame_match = config["Matcher"]["ratio_key_frame_match"].as<double>();
    cfg.ratio_track_local_map = config["Matcher"]["ratio_track_local_map"].as<double>();
    // std::cout << cfg.orb_descriptor_value << " " << cfg.window << " " << cfg.minim_points_found << " " << cfg.ratio_first_second_match << "\n";
    return cfg;
}


Pnp_Ransac_Config load_pnp_ransac_config(const std::string &filename) {
    // PnP RANSAC parameters
    YAML::Node config = YAML::LoadFile(filename);
    Pnp_Ransac_Config cfg;
    cfg.reprojection_window = config["PnP"]["reprojection_window"].as<int>();
    cfg.ransac_iterations = config["PnP"]["ransac_iterations"].as<int>();
    cfg.confidence = config["PnP"]["confidence"].as<double>();
    // std::cout << cfg.reprojection_window << " " << cfg.ransac_iterations << " " << cfg.confidence << "\n";
    return cfg;
}

Config loadConfig(const std::string &filename) {
    YAML::Node config = YAML::LoadFile(filename);
    Config cfg;
    // ORB parameters
    cfg.num_features = config["ORB"]["num_features"].as<int>();
    cfg.split_size = config["ORB"]["split_size"].as<int>();
    cfg.edge_threshold = config["ORB"]["edge_threshold"].as<int>();
    cfg.patch_size = config["ORB"]["patch_size"].as<int>();
    cfg.fast_step = config["ORB"]["fast_step"].as<int>();
    cfg.fast_threshold = config["ORB"]["fast_threshold"].as<int>();
    cfg.min_keypoints_cell = config["ORB"]["min_keypoints_cell"].as<int>();
    cfg.orb_iterations = config["ORB"]["orb_iterations"].as<int>();
    cfg.fast_lower_limit = config["ORB"]["fast_lower_limit"].as<int>();
    cfg.fast_higher_limit = config["ORB"]["fast_higher_limit"].as<int>();
    
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
