#ifndef SLAM_UTILS_H
#define SLAM_UTILS_H
#pragma once
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>

// namespace slam_consts {
//     //tracker
//         const double FOCAL_LENGTH = 132.28;
//         const double X_CAMERA_OFFSET = 110.1;
//         const double Y_CAMERA_OFFSET = 115.7;
//         const double BASELINE = 0.08;
//     //tracker

//     // feature matcher
//     int WINDOW = 16;
//     int EACH_CELL_THRESHOLD = 5;

//     int ORB_EDGE_THRESHOLD = 25;
//     int ORB_ITERATIONS = 10;
//     int FAST_STEP = 5;
//     int FAST_THRESHOLD = 30;
//     int LIMIT_MATCHING = 30;

//     // feature matcher

//     // bundle adjustment
//     int NUMBER_ITERATIONS = 100; 
//     double HUBER_LOSS_VALUE = 1; 
//     ceres::LinearSolverType solver = ceres::LinearSolverType::DENSE_QR;  
//     // bundle adjustment
// };

cv::Mat convert_from_eigen_to_cv2(Eigen::MatrixX<double> matrix);
Eigen::MatrixX<double> convert_from_cv2_to_eigen(cv::Mat matrix);
int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
Eigen::Matrix4d compute_pose_matrix(cv::Mat rotation_matrix, cv::Mat translation);

#endif