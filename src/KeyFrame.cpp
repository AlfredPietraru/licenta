#include "../include/KeyFrame.h"
#include <iostream>

class MapPoint;

KeyFrame::KeyFrame(){};

KeyFrame::KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<cv::KeyPoint> keypoints,
         cv::Mat orb_descriptors, cv::Mat depth_matrix, int idx, cv::Mat frame)
    : Tiw(Tiw), K(K), orb_descriptors(orb_descriptors), depth_matrix(depth_matrix), idx(idx), frame(frame) {
        for (cv::KeyPoint kp : keypoints) {
            this->features.push_back(Feature(kp, this));
        }
    }

Eigen::Vector3d KeyFrame::compute_camera_center() {
    return -this->Tiw.rotationMatrix().transpose() * this->Tiw.translation();
}

Eigen::Vector3d KeyFrame::fromWorldToImage(Eigen::Vector4d& wcoord) {
    Eigen::Vector4d camera_coordinates = this->Tiw.matrix() * wcoord;
    double d = camera_coordinates(2);
    double u = this->K(0, 0) * camera_coordinates(0) / d + this->K(0, 2);
    double v = this->K(1, 1) * camera_coordinates(1) / d + this->K(1, 2);
    return Eigen::Vector3d(u, v, d);
}


float KeyFrame::compute_depth_in_keypoint(cv::KeyPoint kp) {
    // float dd = this->prev_kf->depth_matrix.at<float>(kps[m.queryIdx].pt.y, kps[m.queryIdx].pt.x);
    int x = std::round(kp.pt.x);
    int y = std::round(kp.pt.y);
    uint16_t d = this->depth_matrix.at<uint16_t>(y, x);
    float dd = d / 5000.0f;
    return dd;
}

Eigen::Vector4d KeyFrame::fromImageToWorld(int kp_idx) {
    cv::KeyPoint kp = this->features[kp_idx].kp;
    float dd = this->compute_depth_in_keypoint(kp);
    double new_x = (kp.pt.x - this->K(0, 2)) * dd / this->K(0, 0);
    double new_y = (kp.pt.y - this->K(1, 2)) * dd / this->K(1, 1);
    return this->Tiw.inverse().matrix() *  Eigen::Vector4d(new_x, new_y, dd, 1);
}


std::vector<cv::KeyPoint> KeyFrame::get_all_keypoints() {
    std::vector<cv::KeyPoint> out;
    for (Feature f : this->features) {
        out.push_back(f.get_key_point());
    }
    return out;
}

Eigen::Vector3d KeyFrame::get_viewing_direction() {
    return Tiw.rotationMatrix().col(2).normalized(); 
}

void KeyFrame::correlate_map_points_to_features_current_frame(std::unordered_map<MapPoint *, Feature*>& matches) {
    for (auto it = matches.begin(); it != matches.end(); it++) {
        it->second->set_map_point(it->first);
    }
}

