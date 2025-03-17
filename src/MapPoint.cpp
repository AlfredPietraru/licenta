#include "../include/MapPoint.h"
#include <iostream>

MapPoint::MapPoint(KeyFrame *keyframe, int kp_idx, float depth)
{
    this->belongs_to_keyframes.insert(std::pair<KeyFrame*, int>(keyframe, kp_idx));
    cv::KeyPoint kp = keyframe->features[kp_idx].get_key_point();
    this->wcoord = keyframe->fromImageToWorld(kp_idx);
    Eigen::Vector3d wcoord_local = Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
    this->view_direction = (wcoord_local - keyframe->compute_camera_center()).normalized();
    this->orb_descriptor = keyframe->orb_descriptors.row(kp_idx);
    this->dmax = depth * 1.2; 
    this->dmin = depth * 0.8; 
}

void MapPoint::add_reference_kf(KeyFrame *kf, int idx) {
    this->belongs_to_keyframes.insert(std::pair<KeyFrame*, int>(kf, idx));
}


bool MapPoint::map_point_belongs_to_keyframe(KeyFrame *kf)
{
    // if (this->belongs_to_keyframes.find(kf) != this->belongs_to_keyframes.end()) return true;
    Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(this->wcoord);
    for (int i = 0; i < 3; i++) {
        if (point_camera_coordinates(i) < 0)  return false;
    }
    if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1) return false;
    if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1) return false;
    // correclty reprojected - 1
    // std::cout << "aici";
    if (this->dmax < point_camera_coordinates(2) || this->dmin > point_camera_coordinates(2)) return false;

    // Eigen::Vector3d camera_center = kf->compute_camera_center();
    // Eigen::Vector3d camera_to_point = this->get_3d_vector() - camera_center;
    // Eigen::Vector3d camera_normal = kf->get_viewing_direction();
    // if (camera_to_point.dot(camera_normal) < 0) return false;

    return true;
}

int MapPoint::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}


int MapPoint::find_orb_correspondence(KeyFrame *kf, int window) {
    Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(this->wcoord);
    double u = point_camera_coordinates(0);
    double v = point_camera_coordinates(1);
    int min_hamm_dist = 10000;
    int cur_hamm_dist;
    int right_idx = -1;
    // std::cout << u << " " << v << " ";
    std::vector<cv::KeyPoint> keypoints = kf->get_all_keypoints();
    for (int i = 0; i < keypoints.size(); i++)
    {
        if (keypoints[i].pt.x - window > u || keypoints[i].pt.x + window < u)
            continue;
        if (keypoints[i].pt.y - window > v || keypoints[i].pt.y + window < v)
            continue;
        cur_hamm_dist = ComputeHammingDistance(this->orb_descriptor, kf->orb_descriptors.row(i));
        if (cur_hamm_dist < min_hamm_dist) {
            right_idx = i;
            min_hamm_dist = cur_hamm_dist;
        }
    }
    // std::cout << min_hamm_dist << " ";
    // if (min_hamm_dist == 10000) return -1;
    if (min_hamm_dist > 40) return -1;
    // feature was found - 5
    return right_idx;
}

Eigen::Vector3d MapPoint::get_3d_vector() {
    return Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
}


int MapPoint::reproject_map_point(KeyFrame *kf, int window) {
    int out = this->map_point_belongs_to_keyframe(kf) ?  find_orb_correspondence(kf, window) : -1; 
    return out;
}