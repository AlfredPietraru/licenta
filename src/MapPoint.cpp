#include "../include/MapPoint.h"
#include <iostream>

MapPoint::MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, cv::Mat orb_descriptor,
         int kp_idx, float depth)
{
    this->belongs_to_keyframes.insert({keyframe, kp_idx});
    this->wcoord = wcoord;
    Eigen::Vector3d wcoord_local = Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
    this->view_direction = (wcoord_local - camera_center).normalized();
    this->orb_descriptor = orb_descriptor;
    this->dmax = depth * 1.2; 
    this->dmin = depth * 0.8; 
}

void MapPoint::add_reference_kf(KeyFrame *kf, int idx) {
    this->belongs_to_keyframes.insert({kf, idx});
}

Eigen::Vector3d MapPoint::get_3d_vector() {
    return Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
}
