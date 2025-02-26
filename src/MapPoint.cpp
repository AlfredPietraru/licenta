#include "../include/MapPoint.h"


MapPoint::MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, double depth, Eigen::Matrix4d camera_pose,
                   Eigen::Vector3d camera_center, cv::Mat orb_descriptor)
{
    this->belongs_to_keyframes.insert(keyframe);
    float new_x = (kp.pt.x - X_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
    float new_y = (kp.pt.y - Y_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
    this->wcoord = camera_pose * Eigen::Vector4d(new_x, new_y, depth, 1);
    // this->view_direction = (this->wcoord - camera_center).normalized();
    this->orb_descriptor = orb_descriptor;
    this->dmax = depth * 1.2; // inca nicio idee de ce
    this->dmin = depth * 0.8; // inca nicio idee de ce
}

bool MapPoint::operator==(const MapPoint &lhs)
{
    return (size_t)this == (size_t)&lhs;
}

bool MapPoint::map_point_belongs_to_keyframe(KeyFrame *kf)
{
    if (this->belongs_to_keyframes.find(kf) != this->belongs_to_keyframes.end()) return true;
    float WINDOW = 5;
    std::pair<float, float> camera_coordinates = fromWorldToCamera(kf->Tiw, kf->depth_matrix, this->wcoord);
    if (camera_coordinates.first < 0 || camera_coordinates.second < 0)
        return false;
    float u = camera_coordinates.first;
    float v = camera_coordinates.second;
    int min_hamm_dist = 10000;
    int cur_hamm_dist = -1;
    for (int i = 0; i < kf->keypoints.size(); i++)
    {
        if (kf->keypoints[i].pt.x - WINDOW > u || kf->keypoints[i].pt.x + WINDOW < u)
            continue;
        if (kf->keypoints[i].pt.y - WINDOW > v || kf->keypoints[i].pt.y + WINDOW < v)
            continue;
        cur_hamm_dist = ComputeHammingDistance(this->orb_descriptor, kf->orb_descriptors.row(i));
        min_hamm_dist = cur_hamm_dist < min_hamm_dist ? cur_hamm_dist : min_hamm_dist;
    }
    return cur_hamm_dist != -1;
}