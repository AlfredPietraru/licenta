
#ifndef FEATURE_H
#define FEATURE_H

#include <Eigen/Core>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>

class MapPoint;
class Feature {
public:
    cv::KeyPoint kp;
    cv::KeyPoint kpu;
    cv::Mat descriptor;
    int idx;
    double depth;
    double stereo_depth;
    int curr_hamming_dist = 10000;
    MapPoint *mp = nullptr;

    Feature() {}
    Feature(cv::KeyPoint kp, cv::KeyPoint kpu, cv::Mat descriptor, int idx, double depth, double stereo_depth);
    void set_map_point(MapPoint *mp, int hamming_distance);
    MapPoint* get_map_point();
    cv::KeyPoint get_key_point();
    cv::KeyPoint get_undistorted_keypoint();
    void unmatch_map_point();
    int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
};


#endif