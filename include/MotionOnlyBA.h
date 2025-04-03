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
    BundleAdjustment() {};
    Sophus::SE3d solve(KeyFrame *frame, std::unordered_map<MapPoint *, Feature*>& matches);
};

#endif
