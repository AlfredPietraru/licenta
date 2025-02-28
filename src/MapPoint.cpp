#include "../include/MapPoint.h"
#include <iostream>

MapPoint::MapPoint(KeyFrame *keyframe, int kp_idx)
{
    this->belongs_to_keyframes.insert(std::pair<KeyFrame*, int>(keyframe, kp_idx));
    cv::KeyPoint kp = keyframe->keypoints[kp_idx];
    this->wcoord = keyframe->fromImageToWorld(kp_idx);
    double depth = keyframe->depth_matrix.at<float>((int)kp.pt.x, (int)kp.pt.y);
    // this->view_direction = (this->wcoord - camera_center).normalized();
    this->orb_descriptor = orb_descriptor;
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
    float WINDOW = 7;
    std::pair<float, float> camera_coordinates = kf->fromWorldToImage(this->wcoord);
    if (camera_coordinates.first < 0 || camera_coordinates.second < 0)
        return false;
    float u = camera_coordinates.first;
    float v = camera_coordinates.second;
    int min_hamm_dist = 10000;
    int cur_hamm_dist = 10000;
    for (int i = 0; i < kf->keypoints.size(); i++)
    {
        if (kf->keypoints[i].pt.x - WINDOW > u || kf->keypoints[i].pt.x + WINDOW < u)
            continue;
        if (kf->keypoints[i].pt.y - WINDOW > v || kf->keypoints[i].pt.y + WINDOW < v)
            continue;
        cur_hamm_dist = ComputeHammingDistance(this->orb_descriptor, kf->orb_descriptors.row(i));
        min_hamm_dist = cur_hamm_dist < min_hamm_dist ? cur_hamm_dist : min_hamm_dist;
    }

    return cur_hamm_dist != 10000;
}