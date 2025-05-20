
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
#include "OrbMatcher.h"
#include "config.h"
#include "ORBextractor.h"
#include "ORBVocabulary.h"
#include "MapDrawer.h"
#include "Common.h"
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
    KeyFrame* tracking(Mat frame, Mat depth);
    Tracker(Map *mapp, Config cfg, ORBVocabulary* voc);

    // state parameters of the tracking thread;
    int frames_seen = 0;
    KeyFrame *current_kf = nullptr;
    KeyFrame* prev_kf = nullptr;
    KeyFrame *prev_prev_kf = nullptr;
    KeyFrame *reference_kf = nullptr;
    std::unordered_set<KeyFrame*> local_keyframes;
    Sophus::SE3d velocity = Sophus::SE3d(Eigen::Matrix4d::Identity());

    vector<double> mDistCoef;
    Mat K; 
    Eigen::Matrix3d K_eigen;
    Map *mapp;
    ORBVocabulary* voc;
    MotionOnlyBA *motionOnlyBA;
    OrbMatcher *matcher;
    ORBextractor* extractor = new ORBextractor(1000, 1.2, 8, 20, 7);
    MapDrawer *mapDrawer;

    // tracking time
    int total_tracking_during_matching = 0;
    int total_tracking_during_local_map = 0;
    int motion_only_ba_time = 0;
    int orb_matching_time = 0;

    // tracking constants:
    const int NR_MAP_POINTS_TRACKED_BETWEEN_FRAMES = 20;
    const int NR_MAP_POINTS_TRACKED_MAP_LOW = 30;
    const int NR_MAP_POINTS_TRACKED_MAP_HIGH = 50;
    
    void UndistortKeyPoints(std::vector<cv::KeyPoint>& kps, std::vector<cv::KeyPoint>& u_kps);
    KeyFrame* FindReferenceKeyFrame();
    void TrackReferenceKeyFrame();
    void TrackConsecutiveFrames();
    void TrackLocalMap(Map *mapp);
    bool Is_KeyFrame_needed();
    void GetNextFrame(Mat frame, Mat depth);
    void TrackingWasLost();
    Sophus::SE3d GetVelocityNextFrame();
};

#endif