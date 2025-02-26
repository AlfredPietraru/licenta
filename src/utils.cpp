#include "../include/utils.h"

Eigen::MatrixX<double> convert_from_cv2_to_eigen(cv::Mat matrix) {
    Eigen::MatrixXd out_mat(matrix.size().height, matrix.size().width);
    for (int i = 0; i < matrix.size().height; i++) {
        for (int j = 0; j < matrix.size().width; j++) {
            out_mat(i, j) = matrix.at<double>(i, j);
        }
    }
    return out_mat;
}

Eigen::Vector3d compute_camera_center(Eigen::Matrix4d camera_pose) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i, j) = camera_pose(i, j);
        }
    }
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; i++) {
        t(i) = camera_pose(3, i);
    }
    return -R.transpose() * t;
}

cv::Mat convert_from_eigen_to_cv2(Eigen::MatrixX<double> matrix) {
    cv::Mat out = cv::Mat::zeros(cv::Size(matrix.rows(), matrix.cols()), CV_64FC1);
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            out.at<double>(i,j) = matrix(i, j);
        }
    }
    return out;
}

int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}

std::pair<float, float> fromWorldToCamera(Eigen::Matrix4d &pose, cv::Mat& depth_last_frame, Eigen::Vector4d& wcoord) {
    Eigen::Vector4d camera_coordinates = pose * wcoord;
    if (camera_coordinates(2) <= 1e-6) return std::pair<float, float>(-1, -1);
    double d = camera_coordinates(2);
    float u = FOCAL_LENGTH * camera_coordinates(0) / d + X_CAMERA_OFFSET;
    float v = FOCAL_LENGTH * camera_coordinates(1) / d + Y_CAMERA_OFFSET;
    if (u < 0 || u >= 224) return std::pair<float, float>(-1, -1);
    if (v < 0 || v >= 224) return std::pair<float, float>(-1, -1);
    if (depth_last_frame.at<float>(u, v) < 0) return std::pair<float, float>(-1, -1);
    return std::pair<float, float>(u, v);
}

Eigen::Matrix4d compute_pose_matrix(cv::Mat rotation_matrix, cv::Mat translation) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            T(i,j) = rotation_matrix.at<double>(i, j);
        }
    }
    for (int i = 0; i < 3; i++) {
        T(i, 3) = translation.at<double>(i);
    }
    return T;
}
