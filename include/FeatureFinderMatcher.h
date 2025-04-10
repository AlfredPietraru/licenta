#ifndef FEATURE_MATCHER_FINDER_H
#define FEATURE_MATCHER_FINDER_H

#include <Eigen/Core>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include "config.h"
#include "ORBextractor.h"


class FeatureMatcherFinder {
public:
    int nr_cells_row;
    int nr_cells_collumn;
    int window;
    int orb_edge_threshold;
    int fast_step;
    int orb_iterations;
    int minim_keypoints;
    int fast_lower_limit;
    int fast_higher_limit;
    int fast_threshold;
    int splits;

    std::vector<int> fast_features_cell;
    std::vector<int> nr_keypoints_found;
    std::vector<double> mDistCoef;
    cv::Mat K;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    ORBextractor* extractor = new ORBextractor(1000, 1.2, 8, 20, 7);
    
    FeatureMatcherFinder() {}
    FeatureMatcherFinder(int rows, int cols, Config cfg);

    std::pair<std::pair<std::vector<cv::KeyPoint>, cv::Mat>, std::vector<cv::KeyPoint>> compute_keypoints_descriptors(cv::Mat frame);
private:
    std::vector<cv::KeyPoint> UndistortKeyPoints(std::vector<cv::KeyPoint> kps);
    std::vector<cv::KeyPoint> extract_keypoints(cv::Mat& frame);
};


#endif