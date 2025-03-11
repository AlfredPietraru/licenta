#ifndef BUNDLE_ADJUSTMENT_H
#define BUNDLE_ADJUSTMENT_H

#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "ceres/ceres.h"
#include <sophus/se3.hpp>
#include "MapPoint.h"
#include "KeyFrame.h"
#include <iostream>

class BundleAdjustment
{
public:
    int NUMBER_ITERATIONS = 100; 
    double HUBER_LOSS_VALUE = 2; 
    ceres::LinearSolverType solver = ceres::LinearSolverType::DENSE_QR;  
    
    BundleAdjustment() {}
    Sophus::SE3d solve(KeyFrame *frame, std::vector<MapPoint*> map_points, std::vector<cv::KeyPoint> kps);
};

#endif
