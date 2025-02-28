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
