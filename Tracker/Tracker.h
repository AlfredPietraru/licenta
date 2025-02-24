
#ifndef TRACKING_H
#define TRACKING_H
#include <Eigen/Core>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "../structures.h"


using namespace std;
using namespace cv;

class Tracker {
public:
    std::pair<Graph, vector<MapPoint>>  initialize(Eigen::Matrix3d K);
    void tracking(Eigen::Matrix3d K);

private:
    int LIMIT_MATCHING = 30; 
    KeyFrame last_frame;
    cv::VideoCapture cap = VideoCapture("../video.mp4");
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../fast_depth.onnx");
    Ptr<FastFeatureDetector> fast = cv::FastFeatureDetector::create(60, true);
    Ptr<ORB> orb = ORB::create(500, 1.2F, 8, 30, 0, 2, cv::ORB::HARRIS_SCORE, 5, 20);
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");


    std::pair<cv::Mat, cv::Mat> get_next_image();
    std::pair<std::vector<cv::KeyPoint>, cv::Mat> compute_features_descriptors(cv::Mat frame);
    Eigen::Vector3d compute_camera_center(Eigen::Matrix4d camera_pose);
    int partition(vector<DMatch> &vec, int low, int high);
    void sort_matches_based_on_distance(vector<DMatch> &matches, int low, int high);
    Eigen::Matrix4d estimate_pose(KeyFrame frame, std::vector<DMatch> matches, 
        std::vector<KeyPoint> keypoints_2, Mat depth, Eigen::Matrix3d K);
    cv::Mat convert_from_eigen_to_cv2(Eigen::MatrixX<double> matrix);
};


#endif