#ifndef ORB_MATCHER_CPP
#define ORB_MATCHER_CPP
#include <iostream>
#include <vector>
#include "./MapPoint.h"
#include "./KeyFrame.h"
#include "./Feature.h"
#include "Map.h"

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
    double ratio_key_frame_match;
    double ratio_track_local_map;
    int orb_descriptor_value;
    OrbMatcher(Orb_Matcher orb_matcher_config) {
        this->minim_points_found = orb_matcher_config.minim_points_found;
        this->window = orb_matcher_config.window;
        this->orb_descriptor_value = orb_matcher_config.orb_descriptor_value;
        this->ratio_key_frame_match = orb_matcher_config.ratio_key_frame_match;
        this->ratio_track_local_map = orb_matcher_config.ratio_track_local_map;
    }
    
    void match_frame_reference_frame(std::unordered_map<MapPoint*, Feature*>& matches, KeyFrame *curr, KeyFrame *ref);
    void match_consecutive_frames(std::unordered_map<MapPoint*, Feature*>& matches, KeyFrame* kf, KeyFrame *kf2, int window);
    void match_frame_map_points(std::unordered_map<MapPoint*, Feature*>& matches, KeyFrame *curr, std::unordered_set<MapPoint*>& map_points, int window_size);
    static std::vector<std::pair<int, int>> search_for_triangulation(KeyFrame *ref1, KeyFrame *ref2, Eigen::Matrix3d fundamental_matrix);
    static int ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b); 
    static bool CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const Eigen::Matrix3d &F12);
    static int Fuse(KeyFrame *pKF, KeyFrame *source_kf, const float th);
    void checkOrientation(std::unordered_map<MapPoint*, Feature*>& out, std::unordered_map<MapPoint*, Feature*>& correlation_prev_frame);
};

#endif
