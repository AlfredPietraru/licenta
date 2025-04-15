
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
#include "MapPoint.h"

class Feature {
public:
    int idx;
    cv::KeyPoint kp;
    cv::KeyPoint kpu;
    cv::Mat descriptor;
    int curr_hamming_dist;
    MapPoint *mp;
    double depth;
    double stereo_depth;

    Feature() {}
    Feature(cv::KeyPoint kp, cv::KeyPoint kpu, cv::Mat descriptor, int idx, double depth, double stereo_depth);
    void set_map_point(MapPoint *mp);
    MapPoint* get_map_point();
    cv::KeyPoint get_key_point();
    int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
};


#endif