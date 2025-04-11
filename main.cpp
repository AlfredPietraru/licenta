
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>
#include <filesystem>
#include "include/Tracker.h"
#include "include/LocalMapping.h"
#include "include/TumDatasetReader.h"
#include "include/ORBVocabulary.h"
#include <csignal>
#include <cstdlib>
namespace fs = std::filesystem;
using namespace std;

TumDatasetReader *reader;

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    reader->outfile.close();
    exit(signum);
}

int main(int argc, char **argv)
{
    std::signal(SIGINT, signalHandler);
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
    
    reader = new TumDatasetReader(cfg); 
    std::pair<cv::Mat, cv::Mat> data = reader->get_next_frame();    
    Sophus::SE3d groundtruth_pose = reader->get_next_groundtruth_pose();
    cv::Mat frame = data.first;
    cv::Mat depth = data.second;
    
    Map *mapp = new Map(orb_matcher_cfg);
    LocalMapping *local_mapper = new LocalMapping(mapp);
    Tracker *tracker = new Tracker(cfg, voc, pnp_ransac_cfg, orb_matcher_cfg);
    tracker->initialize(frame, depth, mapp, groundtruth_pose);
    reader->store_entry(groundtruth_pose);
    while(1) {
        std::pair<cv::Mat, cv::Mat> data = reader->get_next_frame();    
        Sophus::SE3d groundtruth_pose = reader->get_next_groundtruth_pose();
        frame = data.first;
        depth = data.second; 
        std::pair<KeyFrame *, bool> tracker_out = tracker->tracking(frame, depth, mapp, groundtruth_pose);
        KeyFrame *kf = tracker_out.first;
        bool needed_keyframe = tracker_out.second;
        reader->store_entry(kf->Tiw);
        if (needed_keyframe) {
            std::cout << "ADAUGA AICI UN KEYFRAME\n";
            local_mapper->local_map(kf);
        }
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
