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
    int orb_descriptor_value;
    double match_reference_frame_orb_descriptor_ratio;
    int DES_DIST_LOW;
    int DES_DIST_HIGH;

    OrbMatcher(Orb_Matcher orb_matcher_config) {
        this->orb_descriptor_value = orb_matcher_config.orb_descriptor_value;
        this->match_reference_frame_orb_descriptor_ratio = orb_matcher_config.match_reference_frame_orb_descriptor_ratio;
        this->DES_DIST_HIGH = orb_matcher_config.des_dist_high;
        this->DES_DIST_LOW = orb_matcher_config.des_dist_low;
    }
    
    void match_frame_reference_frame(KeyFrame *curr, KeyFrame *ref);
    void match_consecutive_frames(KeyFrame* kf, KeyFrame *kf2, int window);
    static std::vector<std::pair<int, int>> search_for_triangulation(KeyFrame *ref1, KeyFrame *ref2, Eigen::Matrix3d fundamental_matrix);
    static int ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b); 
    static bool CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const Eigen::Matrix3d &F12);
    static int Fuse(KeyFrame *pKF, KeyFrame *source_kf, const float th);
    void checkOrientation(std::vector<std::pair<MapPoint*, Feature*>>& current_correlations, KeyFrame *curr, KeyFrame *pref_kf);
};

#endif
