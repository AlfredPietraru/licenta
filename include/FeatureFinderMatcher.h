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
#include "config.h"


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

    std::vector<int> fast_features_cell;
    std::vector<int> nr_keypoints_found;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    FeatureMatcherFinder() {}
    FeatureMatcherFinder(int rows, int cols, Config cfg);

    std::vector<cv::DMatch> match_features_last_frame(KeyFrame *current_kf, KeyFrame *past_kf);
    std::vector<cv::KeyPoint> extract_keypoints(cv::Mat frame);
    cv::Mat compute_descriptors(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints);
};


#endif