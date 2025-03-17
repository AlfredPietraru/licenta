#include "../include/config.h"
#include <iostream>

Config loadConfig(const std::string &filename) {
    YAML::Node config = YAML::LoadFile(filename);
    Config cfg;
    // Tracker
    auto q_yaml = config["Tracker"]["initial_rotation"];
    cfg.initial_rotation = Eigen::Quaterniond(q_yaml[0].as<double>(), q_yaml[1].as<double>(),
             q_yaml[2].as<double>(), q_yaml[3].as<double>());
    cfg.initial_rotation.normalize();
    auto t_yaml = config["Tracker"]["initial_translation"];
    cfg.initial_translation = Eigen::Vector3d(t_yaml[0].as<double>(), t_yaml[1].as<double>(), 
        t_yaml[2].as<double>());
    cfg.initial_pose = Sophus::SE3d(cfg.initial_rotation, cfg.initial_translation);
    
    // ORB parameters
    cfg.num_features = config["ORB"]["num_features"].as<int>();
    cfg.feature_window = config["ORB"]["feature_window"].as<int>();
    cfg.edge_threshold = config["ORB"]["edge_threshold"].as<int>();
    cfg.patch_size = config["ORB"]["patch_size"].as<int>();
    cfg.fast_step = config["ORB"]["fast_step"].as<int>();
    cfg.fast_threshold = config["ORB"]["fast_threshold"].as<int>();
    cfg.min_keypoints_cell = config["ORB"]["min_keypoints_cell"].as<int>();
    cfg.orb_iterations = config["ORB"]["orb_iterations"].as<int>();
    cfg.fast_lower_limit = config["ORB"]["fast_lower_limit"].as<int>();
    cfg.fast_higher_limit = config["ORB"]["fast_higher_limit"].as<int>();
    cfg.interlaping = config["ORB"]["interlaping"].as<int>();
    
    // PnP RANSAC parameters
    cfg.reprojection_window = config["PnP"]["reprojection_window"].as<int>();
    cfg.ransac_iterations = config["PnP"]["ransac_iterations"].as<int>();
    cfg.confidence = config["PnP"]["confidence"].as<double>();

    cfg.orb_descriptor_value = config["Map"]["orb_descriptor_value"].as<int>();
    std::cout << cfg.orb_descriptor_value << "\n\n\n\n";
    
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
