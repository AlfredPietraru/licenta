
#ifndef TRACKING_H
#define TRACKING_H
#include <Eigen/Core>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include "Map.h"
#include "MotionOnlyBA.h"
#include "FeatureFinderMatcher.h"
#include "OrbMatcher.h"
#include "config.h"
#include "ORBVocabulary.h"

using namespace std;
using namespace cv;

class Tracker {
public:
    void initialize(Mat frame, Mat depth, Map &mapp, Sophus::SE3d pose);
    KeyFrame* tracking(Mat frame, Mat depth, Map& map_points, Sophus::SE3d ground_truth_pose);
    Tracker(Config cfg, ORBVocabulary* voc, Pnp_Ransac_Config pnp_ransac_cfg, Orb_Matcher orb_matcher_config) {
        this->K = cfg.K;
        cv::cv2eigen(cfg.K, this->K_eigen);
        this->voc = voc;
        this->mDistCoef = cfg.distortion;
        this->fmf = new FeatureMatcherFinder(480, 640, cfg);
        this->bundleAdjustment = new BundleAdjustment();
        this->matcher = new OrbMatcher(orb_matcher_config);
    }
    KeyFrame *current_kf;
    KeyFrame* prev_kf = nullptr;
    KeyFrame *reference_kf = nullptr;

    vector<double> mDistCoef;
    ORBVocabulary* voc;
    Mat K; 
    Eigen::Matrix3d K_eigen;
    int keyframes_from_last_global_relocalization = 0;
    BundleAdjustment *bundleAdjustment;
    FeatureMatcherFinder *fmf;
    OrbMatcher *matcher;


    
    // important functions
    // Sophus::SE3d TrackWithLastFrame(vector<DMatch> good_matches);
    std::unordered_map<MapPoint*, Feature*> TrackReferenceKeyFrame();
    std::unordered_map<MapPoint*, Feature*> TrackLocalMap(Map &mapp);
    std::unordered_map<MapPoint*, Feature*> TrackConsecutiveFrames();
    bool Is_KeyFrame_needed();
    // auxiliary functions
    void get_current_key_frame(Mat frame, Mat depth);
    void tracking_was_lost();
};

#endif