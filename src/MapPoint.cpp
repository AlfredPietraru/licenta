#include "../include/MapPoint.h"
#include <iostream>

MapPoint::MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, cv::Mat orb_descriptor,
         int kp_idx, float depth)
{
    this->belongs_to_keyframes.insert(std::pair<KeyFrame*, int>(keyframe, kp_idx));
    this->wcoord = wcoord;
    Eigen::Vector3d wcoord_local = Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
    this->view_direction = (wcoord_local - camera_center).normalized();
    this->orb_descriptor = orb_descriptor;
    this->dmax = depth * 1.2; 
    this->dmin = depth * 0.8; 
}

void MapPoint::add_reference_kf(KeyFrame *kf, int idx) {
    this->belongs_to_keyframes.insert(std::pair<KeyFrame*, int>(kf, idx));
}

int inline::MapPoint::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}

int MapPoint::reproject_map_point(KeyFrame *kf, int window, int orb_descriptor_value) {
    Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(this->wcoord);
    for (int i = 0; i < 3; i++) {
        if (point_camera_coordinates(i) < 0)  return -1;
    }
    if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1) return -1;
    if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1) return -1;
    if (this->dmax < point_camera_coordinates(2) || this->dmin > point_camera_coordinates(2)) return -1;
    // map point can be reprojected on image 
    double u = point_camera_coordinates(0);
    double v = point_camera_coordinates(1);
    int min_hamm_dist = 10000;
    int cur_hamm_dist;
    int out = -1;

    std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, window);
    if (kps_idx.size() == 0) return -1;
    for (int idx : kps_idx) {
        cur_hamm_dist = ComputeHammingDistance(this->orb_descriptor, kf->orb_descriptors.row(idx));
        if (cur_hamm_dist < min_hamm_dist) {
            out = idx;
            min_hamm_dist = cur_hamm_dist;
        }
    }


    // for (int i = 0; i < kf->features.size(); i++)
    // {
    //     cv::KeyPoint kp = kf->get_keypoint(i);  
    //     if (kp.pt.x - window > u || kp.pt.x + window < u)
    //         continue;
    //     if (kp.pt.y - window > v || kp.pt.y + window < v)
    //         continue;
    //     cur_hamm_dist = ComputeHammingDistance(this->orb_descriptor, kf->orb_descriptors.row(i));
    //     if (cur_hamm_dist < min_hamm_dist) {
    //         out = i;
    //         min_hamm_dist = cur_hamm_dist;
    //     }
    // }
    if (min_hamm_dist > orb_descriptor_value) return -1;
    return out;
    // feature was found - 5
    // Eigen::Vector3d camera_center = kf->compute_camera_center();
    // Eigen::Vector3d camera_to_point = this->get_3d_vector() - camera_center;
    // Eigen::Vector3d camera_normal = kf->get_viewing_direction();
    // if (camera_to_point.dot(camera_normal) < 0) return false; 
}

Eigen::Vector3d MapPoint::get_3d_vector() {
    return Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
}
