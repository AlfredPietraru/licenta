#include "../include/KeyFrame.h"
#include <iostream>

KeyFrame::KeyFrame(){};

KeyFrame::KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<cv::KeyPoint> keypoints,
         cv::Mat orb_descriptors, cv::Mat depth_matrix, int idx, cv::Mat frame)
    : Tiw(Tiw), K(K), orb_descriptors(orb_descriptors), keypoints(keypoints), depth_matrix(depth_matrix), idx(idx), frame(frame) {}

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
    uint16_t d = this->depth_matrix.at<uint16_t>((int)kp.pt.y, (int)kp.pt.x);
    float dd = d / 5000.0;
    return dd;
}

Eigen::Vector4d KeyFrame::fromImageToWorld(int kp_idx) {
    cv::KeyPoint kp = this->keypoints[kp_idx];
    float dd = this->compute_depth_in_keypoint(kp);
    double new_x = (kp.pt.x - this->K(0, 2)) * dd / this->K(0, 0);
    double new_y = (kp.pt.y - this->K(1, 2)) * dd / this->K(1, 1);
    return this->Tiw.inverse().matrix() *  Eigen::Vector4d(new_x, new_y, dd, 1);
}