
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
    Mat current_frame, current_depth, current_des;
    Eigen::Matrix4d current_T, prev_T;
    std::vector<KeyPoint> current_kps;
    Mat prev_frame, prev_depth, prev_des;
    std::vector<KeyPoint> prev_kps;
    KeyFrame *last_keyframe;


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
    Eigen::Matrix4d TrackWithLastFrame(vector<DMatch> good_matches);
    Eigen::Matrix4d Optimize_Pose_Coordinates(Eigen::Matrix4d& pose, Map mapp);
    bool Is_KeyFrame_needed(Map mapp);

    // auxiliary functions
    vector<DMatch> match_features_last_frame();
    void set_prev_frame();
    void get_next_image();
    void compute_features_descriptors();
    void tracking_was_lost();
};

#endif