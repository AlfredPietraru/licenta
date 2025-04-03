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
#include "config.h"
#include "ORBVocabulary.h"

class OrbMatcher {
public:
    int minim_points_found;
    int window;
    double ration_first_second_match;
    int orb_descriptor_value;
    OrbMatcher(Orb_Matcher orb_matcher_config) {
        this->minim_points_found = orb_matcher_config.minim_points_found;
        this->window = orb_matcher_config.window;
        this->orb_descriptor_value = orb_matcher_config.orb_descriptor_value;
        this->ration_first_second_match = orb_matcher_config.ratio_first_second_match;
    }
    std::unordered_map<MapPoint*, Feature*> match_frame_map_points(KeyFrame* kf, KeyFrame *kf2);
    std::unordered_map<MapPoint*, Feature*> match_frame_map_points(KeyFrame *kf, std::unordered_set<MapPoint*> map_points);
    int get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2);
    int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
    std::unordered_map<MapPoint*, Feature*> checkOrientation(std::unordered_map<MapPoint*, Feature*>& out, std::unordered_map<MapPoint*, Feature*>& correlation_prev_frame);
    std::unordered_map<MapPoint*, Feature*> match_frame_reference_frame(KeyFrame *curr, KeyFrame *ref, ORBVocabulary *voc);
};

#endif
