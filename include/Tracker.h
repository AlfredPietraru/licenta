
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
#include "MapDrawer.h"
#include <chrono>
namespace fs = std::filesystem;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::milliseconds;

using namespace std;
using namespace cv;

class Tracker {
public:
    std::pair<KeyFrame*, bool> tracking(Mat frame, Mat depth, Sophus::SE3d ground_truth_pose);
    Tracker(Mat frame, Mat depth, Map *mapp, Sophus::SE3d pose, Config cfg, 
        ORBVocabulary* voc, Orb_Matcher orb_matcher_config);
    KeyFrame *current_kf = nullptr;
    KeyFrame* prev_kf = nullptr;
    KeyFrame *reference_kf = nullptr;
    Sophus::SE3d velocity = Sophus::SE3d(Eigen::Matrix4d::Identity());

    vector<double> mDistCoef;
    Mat K; 
    Eigen::Matrix3d K_eigen;
    int keyframes_from_last_global_relocalization = 0;
    
    Map *mapp;
    ORBVocabulary* voc;
    MotionOnlyBA *motionOnlyBA;
    FeatureMatcherFinder *fmf;
    OrbMatcher *matcher;
    MapDrawer *mapDrawer;

    int total_tracking_during_matching = 0;
    int total_tracking_during_local_map = 0;


    void TrackReferenceKeyFrame();
    void TrackConsecutiveFrames();
    void TrackLocalMap(Map *mapp);
    bool Is_KeyFrame_needed(Map *mapp, int tracked_by_local_map);
    // auxiliary functions
    void get_current_key_frame(Mat frame, Mat depth);
    void tracking_was_lost();
};

#endif