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
  int des_dist_low;
  int des_dist_high; 
  double ratio_track_local_map;
  double match_reference_frame_orb_descriptor_ratio;
};

struct Config {
  cv::Mat K;
  std::vector<double> distortion;
  Orb_Matcher orb_matcher; 
};

Config loadConfig(const std::string &filename);

#endif 
