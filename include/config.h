#ifndef CONFIGURE_H
#define CONFIGURE_H  

#include <yaml-cpp/yaml.h>
#include <vector>
#include <string>
#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <opencv2/imgproc.hpp>

struct Orb_Matcher {
  int orb_descriptor_value;
  int window;
  int minim_points_found;
  double ratio_key_frame_match;
  double ratio_track_local_map;
};

struct Pnp_Ransac_Config {
    int reprojection_window;
    int ransac_iterations;
    float confidence;
};

struct Config {
  cv::Mat K;
  std::vector<double> distortion;
  Pnp_Ransac_Config pnp_ransac_config;
  Orb_Matcher orb_matcher; 
};

Config loadConfig(const std::string &filename);

#endif 
