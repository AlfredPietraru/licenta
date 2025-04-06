
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>
#include <filesystem>
#include "include/Tracker.h"
#include "include/TumDatasetReader.h"
#include "include/ORBVocabulary.h"
namespace fs = std::filesystem;
using namespace std;

int main(int argc, char **argv)
{
    ORBVocabulary *voc = new ORBVocabulary();
    bool bVocLoad = voc->loadFromTextFile("../ORBvoc.txt");
    if (!bVocLoad) {
        std::cout << "Nu s-a putut incarca corespunzator fisierul ORBvoc.txt\n";
        exit(1);
    } else {
        std::cout << "Fisierul a fost incarcat cu succes\n";
    }
    Config cfg = loadConfig("../config.yaml");
    Pnp_Ransac_Config pnp_ransac_cfg = load_pnp_ransac_config("../config.yaml");
    Orb_Matcher orb_matcher_cfg = load_orb_matcher_config("../config.yaml");

    
    TumDatasetReader *reader = new TumDatasetReader(cfg); 
    std::pair<std::pair<cv::Mat, cv::Mat>, Sophus::SE3d> data = reader->get_next_frame();    
    cv::Mat frame = data.first.first;
    cv::Mat depth = data.first.second;
    Sophus::SE3d pose = data.second;
    
    Map mapp = Map(orb_matcher_cfg);
    Tracker *tracker = new Tracker(cfg, voc, pnp_ransac_cfg, orb_matcher_cfg);
    tracker->initialize(frame, depth, mapp, pose);
    while(1) {
        std::pair<std::pair<cv::Mat, cv::Mat>, Sophus::SE3d> data = reader->get_next_frame();
        frame = data.first.first;
        depth = data.first.second;
        pose = data.second; 
        tracker->tracking(frame, depth, mapp, pose);
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
