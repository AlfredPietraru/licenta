#include "../include/MapPoint.h"



MapPoint::MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, 
        cv::Mat orb_descriptor, int kp_idx)
{
    this->belongs_to_keyframes.insert({keyframe, kp_idx});
    this->wcoord = wcoord;
    this->wcoord_3d = Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
    this->view_direction = (wcoord_3d - camera_center);
    double distance = (this->view_direction).norm();
    this->view_direction.normalize();
    this->orb_descriptor = orb_descriptor;
    this->dmax = distance * pow(1.2, kp.octave);
    this->dmin = this->dmax / pow(1.2, 8);
    this->is_safe_to_use = true;
    // this->is_safe_to_use = depth < BASELINE * 40 && depth > 0;
}

void MapPoint::add_reference_kf(KeyFrame *kf, int idx) {
    this->belongs_to_keyframes.insert({kf, idx});
}

Eigen::Vector3d MapPoint::get_3d_vector() {
    return Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
}


int MapPoint::check_index_in_keyframe(KeyFrame *kf) {
    if (this->belongs_to_keyframes.find(kf) != this->belongs_to_keyframes.end()) {
        return this->belongs_to_keyframes[kf];
    }
    return -1;
}

int MapPoint::predict_image_scale(double distance) {
    float ratio = this->dmax / distance;
    int scale = ceil(log(ratio) / log(1.2));
    scale = (scale < 0) ? 0 : scale;
    scale = (scale >= 8) ? scale - 1 : scale;
    return scale;
}
