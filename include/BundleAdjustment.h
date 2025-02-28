#ifndef BUNDLE_ADJUSTMENT_H
#define BUNDLE_ADJUSTMENT_H

#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "ceres/ceres.h"
#include <sophus/se3.hpp>
#include "utils.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <iostream>

class BundleAdjustment
{
public:
    std::vector<cv::KeyPoint> kps;
    Sophus::SE3d T;
    std::vector<MapPoint*> map_points;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    double WINDOW = 7;
    

    BundleAdjustment() {}
    BundleAdjustment(std::vector<MapPoint*> map_points, KeyFrame *frame);
    Sophus::SE3d solve();
};

#endif
