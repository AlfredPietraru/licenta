#ifndef SLAM_UTILS_H
#define SLAM_UTILS_H
#pragma once
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "structures.h"

Eigen::Vector3d compute_camera_center(Eigen::Matrix4d camera_pose);
cv::Mat convert_from_eigen_to_cv2(Eigen::MatrixX<double> matrix);
Eigen::MatrixX<double> convert_from_cv2_to_eigen(cv::Mat matrix);
int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
std::pair<float, float> fromWorldToCamera(Eigen::Matrix4d &camera_pose, cv::Mat& depth_last_frame, Eigen::Vector4d& wcoord);

#endif