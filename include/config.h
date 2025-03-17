#ifndef CONFIGURE_H
#define CONFIGURE_H  

#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/imgproc.hpp>

struct Config {
    cv::Mat K;
    std::vector<double> distortion;
    Eigen::Quaterniond initial_rotation;
    Eigen::Vector3d initial_translation;
    Sophus::SE3d initial_pose;

    // ORB
    int num_features;
    // dimensiunea ferestrei in care gasim keypoints
    int feature_window;
    // marginea pe care o respecta feature-urile orb extrase, sa nu fie aproape de margine
    int edge_threshold;
    // cat de mare sa fie zona gasita de keypoint-uri
    int patch_size;
    // cate keypoint-uri minim per celula
    int min_keypoints_cell;
    // de cate ori sa itereze algoritmul orb
    int orb_iterations;
    int fast_step;
    int fast_lower_limit;
    int fast_higher_limit;
    int fast_threshold;
    int interlaping;

    int reprojection_window;
    int ransac_iterations;
    float confidence;
    int orb_descriptor_value;
};

Config loadConfig(const std::string &filename);

#endif 
