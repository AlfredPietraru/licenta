
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

// FORWARD DECLARATION
class KeyFrame;
class MapPoint;

class Feature {
public:
    cv::KeyPoint kp;
    KeyFrame *frame;
    MapPoint *mp;

    Feature() {}
    Feature(cv::KeyPoint kp, KeyFrame *frame, MapPoint *mp);
    Feature(cv::KeyPoint kp, KeyFrame *frame);
    void set_map_point(MapPoint *mp);
    void set_key_frame(KeyFrame *frame);
    MapPoint* get_map_point();
    KeyFrame* get_key_frame();
    cv::KeyPoint get_key_point();
};


#endif