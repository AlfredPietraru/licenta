#ifndef MAP_STRUCTURE_H
#define MAP_STRUCTURE_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include "structures.h"
#include "utils.h"

class Map {
public:
    std::vector<MapPoint*> map_points;

    Map();

    void add_multiple_map_points(KeyFrame frame, std::vector<MapPoint*> new_points);
    std::vector<MapPoint*> get_map_points(KeyFrame frame);
    std::vector<MapPoint*> return_map_points_seen_in_frame(Eigen::Matrix4d& pose, cv::Mat &depth);
};

#endif
