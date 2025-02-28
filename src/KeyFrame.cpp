#include "../include/KeyFrame.h"
#include <iostream>

KeyFrame::KeyFrame(){};

KeyFrame::KeyFrame(Eigen::Matrix4d Tiw, Eigen::Matrix3d intrisics, std::vector<cv::KeyPoint> keypoints,
         cv::Mat orb_descriptors, cv::Mat depth_matrix)
    : Tiw(Tiw), intrisics(intrisics), orb_descriptors(orb_descriptors), keypoints(keypoints), depth_matrix(depth_matrix) {}

bool KeyFrame::operator==(const KeyFrame &lhs)
{
    return (size_t)this == (size_t)&lhs;
}

Eigen::Vector3d KeyFrame::compute_camera_center() {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i, j) = this->Tiw(i, j);
        }
    }
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; i++) {
        t(i) = this->Tiw(3, i);
    }
    return -R.transpose() * t;
}


std::pair<float, float> KeyFrame::fromWorldToImage(Eigen::Vector4d& wcoord) {
    Eigen::Vector4d camera_coordinates = this->Tiw * wcoord;
    double d = camera_coordinates(2);
    float u = this->intrisics(0, 0) * camera_coordinates(0) / d + this->intrisics(0, 2);
    float v = this->intrisics(1, 1) * camera_coordinates(1) / d + this->intrisics(1, 2);
    if (u < 0 || u >= 224) return std::pair<float, float>(-1, -1);
    if (v < 0 || v >= 224) return std::pair<float, float>(-1, -1);
    if (this->depth_matrix.at<float>((int)u, (int)v) < 0) return std::pair<float, float>(-1, -1);
    return std::pair<float, float>(u, v);
}

Eigen::Vector4d KeyFrame::fromImageToWorld(int kp_idx) {
    cv::KeyPoint kp = this->keypoints[kp_idx];
    float d = depth_matrix.at<float>((int)kp.pt.y, (int)kp.pt.x);
    float new_x = (kp.pt.x - this->intrisics(0, 2)) * d / this->intrisics(0, 0);
    float new_y = (kp.pt.y - this->intrisics(1, 2)) * d / this->intrisics(1, 1);
    return this->Tiw.inverse() * Eigen::Vector4d(new_x, new_y, d, 1);
}