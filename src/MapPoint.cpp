#include "../include/MapPoint.h"



MapPoint::MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, cv::Mat orb_descriptor,
         int kp_idx, float depth)
{
    this->belongs_to_keyframes.insert({keyframe, kp_idx});
    this->wcoord = wcoord;
    this->wcoord_3d = Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
    this->view_direction = (wcoord_3d - camera_center);
    this->view_direction.normalize();
    this->orb_descriptor = orb_descriptor;
    this->dmax = depth * 1.2; 
    this->dmin = depth * 0.8;
    this->is_safe_to_use = depth < BASELINE * 40 && depth > 0;
}

void MapPoint::add_reference_kf(KeyFrame *kf, int idx) {
    this->belongs_to_keyframes.insert({kf, idx});
}

Eigen::Vector3d MapPoint::get_3d_vector() {
    return Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
}
