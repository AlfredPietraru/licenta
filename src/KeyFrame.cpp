#include "../include/KeyFrame.h"

KeyFrame::KeyFrame(){};

KeyFrame::KeyFrame(Eigen::Matrix4d Tiw, Eigen::Matrix3d intrisics, std::vector<cv::KeyPoint> keypoints,
         cv::Mat orb_descriptors, cv::Mat depth_matrix)
    : Tiw(Tiw), intrisics(intrisics), orb_descriptors(orb_descriptors), keypoints(keypoints), depth_matrix(depth_matrix) {}

bool KeyFrame::operator==(const KeyFrame &lhs)
{
    return (size_t)this == (size_t)&lhs;
}

bool KeyFrame::map_point_belongs_to_keyframe(MapPoint *mp)
{
    float WINDOW = 5;
    std::pair<float, float> camera_coordinates = fromWorldToCamera(this->Tiw, this->depth_matrix, mp->wcoord);
    if (camera_coordinates.first < 0 || camera_coordinates.second < 0)
        return false;
    float u = camera_coordinates.first;
    float v = camera_coordinates.second;
    int min_hamm_dist = 10000;
    int cur_hamm_dist = -1;
    for (int i = 0; i < this->keypoints.size(); i++)
    {
        if (this->keypoints[i].pt.x - WINDOW > u || this->keypoints[i].pt.x + WINDOW < u)
            continue;
        if (this->keypoints[i].pt.y - WINDOW > v || this->keypoints[i].pt.y + WINDOW < v)
            continue;
        cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, this->orb_descriptors.row(i));
        min_hamm_dist = cur_hamm_dist < min_hamm_dist ? cur_hamm_dist : min_hamm_dist;
    }
    return cur_hamm_dist != -1;
}