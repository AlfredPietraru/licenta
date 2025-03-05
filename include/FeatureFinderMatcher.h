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


class FeatureMatcherFinder {
public:
    int window;
    int nr_cells;
    int INITIAL_FAST_INDEX = 50;
    int EACH_CELL_THRESHOLD = 5;
    int FAST_ITERATIONS = 10;
    int ORB_ITERATIONS = 5;
    int FAST_STEP = 5;

    std::vector<std::vector<cv::Ptr<cv::FastFeatureDetector>>> fast_vector;
    cv::Ptr<cv::ORB> orb;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    
    FeatureMatcherFinder() {}
    FeatureMatcherFinder(cv::Size frame_size, int window_size);

    std::vector<cv::KeyPoint> extract_keypoints(cv::Mat frame);
    cv::Mat compute_descriptors(cv::Mat frame, std::vector<cv::KeyPoint>& keypoints);
};


#endif