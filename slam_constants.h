#include <iostream>
#include <opencv2/core/core.hpp>

#ifndef my_constants
#define my_constants
    const int NR_FEATURES_THRESHOLD = 1000;
    const int NR_FEATURES_MATHCED = 100;
    const std::string DESCRIPTION_MATCHER_ALGORITHM = std::string("BruteForce-Hamming");
    const cv::Mat CAMERA_MATRIX = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);  
    const float FOCAL_LENGTH = CAMERA_MATRIX.at<double>(1, 1); 
    const cv::Point2d PRINCIPAL_POINT = cv::Point2d(CAMERA_MATRIX.at<float>(0, 2), CAMERA_MATRIX.at<float>(1, 2));
    const double DEPTH_NORMALIZATION = 5000.0;
#endif