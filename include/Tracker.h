
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
#include "utils.h"
#include "Graph.h"
#include "BundleAdjustment.h"
using namespace std;
using namespace cv;

class Tracker {
public:
    std::pair<Graph, Map> initialize();
    void tracking(Map map_points, vector<KeyFrame*> &key_frames_buffer);
    Tracker(){
        double data_vector[9] = {520.9, 0.0, 325.1, 0.0, 521.0, 249.7, 0.0, 0.0, 1.0};
        cv::Mat K_cv = cv::Mat(3, 3, CV_64F, data_vector);
        this->K = convert_from_cv2_to_eigen(K_cv);
    }

private:
    KeyFrame *current_kf;
    KeyFrame* prev_kf = nullptr;
    KeyFrame *reference_kf;


    int last_keyframe_added = 0;
    int keyframes_from_last_global_relocalization = 0;



    int LIMIT_MATCHING = 30; 
    int WINDOW = 5;
    Eigen::Matrix3d K; 
    cv::VideoCapture cap = cv::VideoCapture("../video.mp4");
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../fast_depth.onnx");
    cv::Ptr<cv::FastFeatureDetector> fast = cv::FastFeatureDetector::create(60, true);
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2F, 8, 30, 0, 2, cv::ORB::HARRIS_SCORE, 5, 20);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
    BundleAdjustment bundleAdjustment;
    
    // important functions
    Sophus::SE3d TrackWithLastFrame(vector<DMatch> good_matches);
    void Optimize_Pose_Coordinates(Map mapp);
    bool Is_KeyFrame_needed(Map mapp);

    // auxiliary functions
    vector<DMatch> match_features_last_frame();
    void set_prev_key_frame();
    void get_current_key_frame();
    void tracking_was_lost();
};

#endif