
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
#include <sophus/se3.hpp>
#include <opencv2/core/eigen.hpp>
#include "Map.h"
#include "BundleAdjustment.h"
#include "FeatureFinderMatcher.h"
#include "OrbMatcher.h"
#include "config.h"

using namespace std;
using namespace cv;

class Tracker {
public:
    Map initialize(Mat frame, Mat depth, Config cfg);
    void tracking(Mat frame, Mat depth, Map map_points, vector<KeyFrame*> &key_frames_buffer);
    Tracker(Config cfg) {
        this->K = cfg.K;
        cv::cv2eigen(cfg.K, this->K_eigen);
        this->initial_pose = cfg.initial_pose;
        this->fmf = new FeatureMatcherFinder(480, 640, cfg);
        this->bundleAdjustment = new BundleAdjustment();
        std::cout << cfg.orb_descriptor_value << "\n\n";
        this->matcher = new OrbMatcher(cfg.orb_descriptor_value);
        this->optimizer_window = cfg.reprojection_window;
        this->ransac_iteration = cfg.ransac_iterations;
        this->ransac_confidence = cfg.confidence;
    }

private:
    KeyFrame *current_kf;
    KeyFrame* prev_kf = nullptr;
    KeyFrame *reference_kf = nullptr;
    Mat K; 
    Eigen::Matrix3d K_eigen;
    Sophus::SE3d initial_pose;
    int optimizer_window;
    int ransac_iteration;
    float ransac_confidence;


    int frames_tracked = 0;
    int last_keyframe_added = 0;
    int keyframes_from_last_global_relocalization = 0;

    BundleAdjustment *bundleAdjustment;
    FeatureMatcherFinder *fmf;
    OrbMatcher *matcher;

    
    // important functions
    // Sophus::SE3d TrackWithLastFrame(vector<DMatch> good_matches);
    Sophus::SE3d TrackWithLastFrame(std::vector<std::pair<MapPoint*, cv::KeyPoint>> matches, std::vector<int> &inliers);
    void Optimize_Pose_Coordinates(Map mapp, std::vector<std::pair<MapPoint*, cv::KeyPoint>> matches, vector<int> inliers);
    bool Is_KeyFrame_needed(Map mapp);
    void VelocityEstimation();


    // auxiliary functions
    void get_current_key_frame(Mat frame, Mat depth);
    void tracking_was_lost();
    void reject_outlier(std::unordered_map<MapPoint *, cv::KeyPoint>& matches, std::unordered_map<MapPoint *, cv::KeyPoint>& inliers);
    std::unordered_map<MapPoint *, cv::KeyPoint> get_outliers(std::vector<std::pair<MapPoint *, cv::KeyPoint>>& matches, vector<int> inliers);
};

#endif