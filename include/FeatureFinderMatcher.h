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
#include "KeyFrame.h"


class FeatureMatcherFinder {
public:
    // int WINDOW = 32;
    int nr_cells_row;
    int nr_cells_collumn;
    int WINDOW = 40;
    int EACH_CELL_THRESHOLD = 5;
    int ORB_FEATURES = 500;
    int ORB_EDGE_THRESHOLD = 25;
    int ORB_ITERATIONS = 10;
    int FAST_STEP = 5;
    int FAST_THRESHOLD = 30;

    int LOWER_LIMIT_THRESHOLD = 10;
    int HIGH_LIMIT_THRESHOLD = 60;
    
    int LIMIT_MATCHING = 10;

    std::vector<int> fast_features_cell;
    std::vector<int> nr_keypoints_found;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    cv::Mat mask;
    
    FeatureMatcherFinder() {}
    FeatureMatcherFinder(cv::Mat frame);

    std::vector<cv::DMatch> match_features_last_frame(KeyFrame *current_kf, KeyFrame *past_kf);
    std::vector<cv::KeyPoint> extract_keypoints(cv::Mat frame);
    cv::Mat compute_descriptors(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints);
};


#endif