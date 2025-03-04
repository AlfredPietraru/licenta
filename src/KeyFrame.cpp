#include "../include/KeyFrame.h"
#include <iostream>

KeyFrame::KeyFrame(){};

KeyFrame::KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d intrisics, std::vector<cv::KeyPoint> keypoints,
         cv::Mat orb_descriptors, cv::Mat depth_matrix)
    : Tiw(Tiw), intrisics(intrisics), orb_descriptors(orb_descriptors), keypoints(keypoints), depth_matrix(depth_matrix) {}

bool KeyFrame::operator==(const KeyFrame &lhs)
{
    return (size_t)this == (size_t)&lhs;
}

Eigen::Vector3d KeyFrame::compute_camera_center() {
    return -this->Tiw.rotationMatrix().transpose() * this->Tiw.translation();
}


Eigen::Vector3d KeyFrame::fromWorldToImage(Eigen::Vector4d& wcoord) {
    Eigen::Vector4d camera_coordinates = this->Tiw * wcoord;
    float d = camera_coordinates(2);
    float u = this->intrisics(0, 0) * camera_coordinates(0) / d + this->intrisics(0, 2);
    float v = this->intrisics(1, 1) * camera_coordinates(1) / d + this->intrisics(1, 2);
    return Eigen::Vector3d(u, v, d);
}

Eigen::Vector4d KeyFrame::fromImageToWorld(int kp_idx) {
    cv::KeyPoint kp = this->keypoints[kp_idx];
    float d = depth_matrix.at<float>((int)kp.pt.x, (int)kp.pt.y);
    float new_x = (kp.pt.x - this->intrisics(0, 2)) * d / this->intrisics(0, 0);
    float new_y = (kp.pt.y - this->intrisics(1, 2)) * d / this->intrisics(1, 1);
    return this->Tiw.inverse() *  Eigen::Vector4d(new_x, new_y, d, 1);
}