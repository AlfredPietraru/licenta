#include "../include/KeyFrame.h"

KeyFrame::KeyFrame(){};

KeyFrame::KeyFrame(Eigen::Matrix4d Tiw, Eigen::Matrix3d intrisics, std::vector<cv::KeyPoint> keypoints,
         cv::Mat orb_descriptors, cv::Mat depth_matrix)
    : Tiw(Tiw), intrisics(intrisics), orb_descriptors(orb_descriptors), keypoints(keypoints), depth_matrix(depth_matrix) {}

bool KeyFrame::operator==(const KeyFrame &lhs)
{
    return (size_t)this == (size_t)&lhs;
}