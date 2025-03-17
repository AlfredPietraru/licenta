// #include "DBoW3/DBoW3.h"
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>
#include <filesystem>
#include "include/Tracker.h"

namespace fs = std::filesystem;

using namespace std;

int main(int argc, char **argv)
{
    std::string rgb_path = "../rgbd_dataset_freiburg1_xyz/rgb";
    std::string depth_path = "../rgbd_dataset_freiburg1_xyz/depth";
    Config cfg = loadConfig("../config.yaml");
    Tracker *tracker = new Tracker(cfg);
    bool start = true;
    Map mapp;
    Mat distorted_frame, frame, depth; 
    vector<KeyFrame*> keyframes_buffer;
    vector<std::string> rgb_file_paths;
    vector<std::string> depth_file_paths;
    for (std::filesystem::__cxx11::directory_entry entry : fs::directory_iterator(rgb_path)) {
        rgb_file_paths.push_back(entry.path());
    }
    for (std::filesystem::__cxx11::directory_entry entry : fs::directory_iterator(depth_path)) {
        depth_file_paths.push_back(entry.path());
    }
    
    std::sort(rgb_file_paths.begin(), rgb_file_paths.end());
    std::sort(depth_file_paths.begin(), depth_file_paths.end());
    for (int i = 0; i < rgb_file_paths.size(); i++) {
        std::cout << rgb_file_paths[i] << " " << depth_file_paths[i] << "\n";
        distorted_frame = cv::imread(rgb_file_paths[i], cv::IMREAD_COLOR_RGB);
        depth = cv::imread(depth_file_paths[i], cv::IMREAD_UNCHANGED);
        // cv::imshow("distorted_image", distorted_frame);
        // cv::imshow("Depth Map", depth);
        // cv::waitKey(0);
        cv::undistort(distorted_frame, frame, cfg.K, cfg.distortion);
        if (start) {
            mapp = tracker->initialize(frame, depth);
            start = !start;
            continue;
        }
        tracker->tracking(frame, depth, mapp, keyframes_buffer);
    }
}


// int read_img() {
//     cv::VideoCapture cap = cv::VideoCapture("../video.mp4");
//     cv::dnn::Net net = cv::dnn::readNetFromONNX("../fast_depth.onnx");
//     cv::Mat K = cv::Mat::eye(cv::Size(3, 3), CV_32F);
//     cv::Size size = cv::Size(224, 224);
//     K.at<float>(0, 0) = 1436.71;
//     K.at<float>(1, 1) = 1436.71;
//     K.at<float>(0, 2) = 757.49;
//     K.at<float>(1, 2) = 1017.88;
//     // std::cout << K << "\n";

//     cv::Mat frame, distorted_frame, resized, normalized_frame;
//     cap.read(distorted_frame);
//     std::cout << distorted_frame.size() << "\n";
//     vector<double> distorsions = {2.04832995e-01, -8.63937346e-01, -9.53701444e-04, -1.13478117e-03, 1.60886073e+00};
//     cv::undistort(distorted_frame, frame, K, distorsions);
//     cv::resize(frame, resized, size);
//     cv::normalize(resized, normalized_frame, 0, 255, cv::NORM_MINMAX);
//     Mat blob = cv::dnn::blobFromImage(normalized_frame, 1, size, cv::Scalar(), true, false);
//     net.setInput(blob);
//     Mat depth = net.forward().reshape(1, 224);
//     depth *= 10;
//     K.at<float>(0, 0) = 1436.71 * (size.height / (double)distorted_frame.size().height);
//     K.at<float>(1, 1) = 1436.71 * (size.width / (double)distorted_frame.size().width);
//     K.at<float>(0, 2) = 757.49 * (size.height / (double)distorted_frame.size().height);
//     K.at<float>(1, 2) = 1017.88 * (size.width / (double)distorted_frame.size().width);
//     std::cout << K << "\n";
//     Sophus::SE3d initial_pose = Sophus::SE3d(Eigen::Matrix4d::Identity());
//     Tracker *tracker = new Tracker(K, initial_pose, 10);
//     Map mapp = tracker->initialize(resized, depth);
//     vector<KeyFrame*> keyframes_buffer;
//     while(1) {
//         cap.read(distorted_frame);
//         cv::undistort(distorted_frame, frame, K, distorsions);
//         cv::resize(frame, resized, size);
//         cv::normalize(resized, normalized_frame, 0, 255, cv::NORM_MINMAX);
//         blob = cv::dnn::blobFromImage(normalized_frame, 1, size, cv::Scalar(), true, false);
//         net.setInput(blob);
//         depth = net.forward().reshape(1, 224);
//         depth *= 10;
//         tracker->tracking(resized, depth, mapp, keyframes_buffer);
//     }
//     return 0;
// }
