#ifndef ORB_MATCHER_CPP
#define ORB_MATCHER_CPP
#include <iostream>
#include <vector>
#include "./MapPoint.h"
#include "./KeyFrame.h"
#include "./Feature.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

class OrbMatcher {
public:
    int minim_points_found;
    int window;
    int orb_descriptor_value;
    OrbMatcher(int minim_points_found, int window, int orb_descriptor_value) : minim_points_found(minim_points_found), window(window), 
        orb_descriptor_value(orb_descriptor_value) {}
    std::vector<std::pair<MapPoint*, Feature*>> match_two_consecutive_frames(KeyFrame *prev_kf, KeyFrame *curr_kf); 
};

#endif
