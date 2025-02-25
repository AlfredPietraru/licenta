#ifndef BUNDLE_ADJUSTMENT_H
#define BUNDLE_ADJUSTMENT_H

#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "ceres/ceres.h"
#include "../structures.h"
#include <iostream>

class BundleAdjustment
{
public:
    std::vector<cv::KeyPoint> kps;
    double data_T[12];
    Eigen::Matrix4d T;
    std::vector<MapPoint> map_points;
    std::vector<double*> poins_coordinates;
    

    BundleAdjustment() {}
    BundleAdjustment(std::vector<MapPoint> map_points, std::vector<cv::KeyPoint> kps, Eigen::Matrix4d T);
    Eigen::Matrix4d return_optimized_pose();
    void solve();
};

#endif
