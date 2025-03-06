#include "../include/MapPoint.h"
#include <iostream>

MapPoint::MapPoint(KeyFrame *keyframe, int kp_idx)
{
    this->belongs_to_keyframes.insert(std::pair<KeyFrame*, int>(keyframe, kp_idx));
    cv::KeyPoint kp = keyframe->keypoints[kp_idx];
    this->wcoord = keyframe->fromImageToWorld(kp_idx);
    double depth = keyframe->depth_matrix.at<float>((int)kp.pt.x, (int)kp.pt.y);
    // this->view_direction = (this->wcoord - camera_center).normalized();
    this->orb_descriptor = keyframe->orb_descriptors.row(kp_idx);
    this->dmax = depth * 1.2; // inca nicio idee de ce
    this->dmin = depth * 0.8; // inca nicio idee de ce
}

bool MapPoint::operator==(const MapPoint &lhs)
{
    return (size_t)this == (size_t)&lhs;
}

void MapPoint::add_reference_kf(KeyFrame *kf, int idx) {
    this->belongs_to_keyframes.insert(std::pair<KeyFrame*, int>(kf, idx));
}


bool MapPoint::map_point_belongs_to_keyframe(KeyFrame *kf)
{
    if (this->belongs_to_keyframes.find(kf) != this->belongs_to_keyframes.end()) return true;
    Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(this->wcoord);
    for (int i = 0; i < 3; i++) {
        if (point_camera_coordinates(i) < 0)  return false;
    }
    // correclty reprojected - 1
    if (this->dmax < point_camera_coordinates(2) || this->dmin > point_camera_coordinates(2)) return false;
    // distance match values - 3
    float u = point_camera_coordinates(0);
    float v = point_camera_coordinates(1);
    int min_hamm_dist = 10000;
    int cur_hamm_dist;
    for (int i = 0; i < kf->keypoints.size(); i++)
    {
        if (kf->keypoints[i].pt.x - this->WINDOW > u || kf->keypoints[i].pt.x + this->WINDOW < u)
            continue;
        if (kf->keypoints[i].pt.y - this->WINDOW > v || kf->keypoints[i].pt.y + this->WINDOW < v)
            continue;
        cur_hamm_dist = ComputeHammingDistance(this->orb_descriptor, kf->orb_descriptors.row(i));
        min_hamm_dist = cur_hamm_dist < min_hamm_dist ? cur_hamm_dist : min_hamm_dist;
    }
    if (min_hamm_dist == 10000) return false;
    // feature was found - 5
    return true;
}