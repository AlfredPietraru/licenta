
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
#include "BundleAdjustment.h"
#include "FeatureFinderMatcher.h"
using namespace std;
using namespace cv;

class Tracker {
public:
    Map initialize(Mat frame, Mat depth);
    void tracking(Mat frame, Mat depth, Map map_points, vector<KeyFrame*> &key_frames_buffer);
    Tracker(){
        // 124.72440945
        double data_vector[9] = {132.28, 0.0, 110.1, 0.0, 132.28, 115.7, 0.0, 0.0, 1.0};
        // double data_vector[9] = {525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0};
        cv::Mat K_cv = cv::Mat(3, 3, CV_64F, data_vector);
        this->K = convert_from_cv2_to_eigen(K_cv);
    }

private:
    KeyFrame *current_kf;
    KeyFrame* prev_kf = nullptr;
    KeyFrame *reference_kf;


    int frames_tracked = 0;
    int last_keyframe_added = 0;
    int keyframes_from_last_global_relocalization = 0;



    int LIMIT_MATCHING = 30; 
    int WINDOW = 5;
    Eigen::Matrix3d K; 
    BundleAdjustment *bundleAdjustment = new BundleAdjustment();
    FeatureMatcherFinder *fmf;

    
    // important functions
    Sophus::SE3d TrackWithLastFrame(vector<DMatch> good_matches);
    void Optimize_Pose_Coordinates(Map mapp);
    bool Is_KeyFrame_needed(Map mapp);

    // auxiliary functions
    void get_current_key_frame(Mat frame, Mat depth);
    void tracking_was_lost();
};

#endif