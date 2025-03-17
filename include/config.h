#ifndef CONFIGURE_H
#define CONFIGURE_H  

#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <opencv2/imgproc.hpp>

struct Config {
    int reprojection_window;
    int num_features;
    float scale_factor;
    int num_levels;
    int ransac_iterations;
    double reprojection_error;
    cv::Mat K;
    std::vector<double> distortion;
};

Config loadConfig(const std::string &filename);

#endif 
