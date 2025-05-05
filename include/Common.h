#ifndef COMMON_SLAM_H
#define COMMON_SLAM_H

#include "Map.h"

class Common {
public:
    static double get_rgbd_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* feature, double chi2);
    static double get_monocular_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* feature, double chi2);
};


#endif