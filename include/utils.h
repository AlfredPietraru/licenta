#ifndef SLAM_UTILS_H
#define SLAM_UTILS_H
#pragma once
#include <Eigen/Core>
#include <opencv2/core/core.hpp>

cv::Mat convert_from_eigen_to_cv2(Eigen::MatrixX<double> matrix);
Eigen::MatrixX<double> convert_from_cv2_to_eigen(cv::Mat matrix);
int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
Eigen::Matrix4d compute_pose_matrix(cv::Mat rotation_matrix, cv::Mat translation);

#endif