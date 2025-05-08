
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
#include <chrono>
namespace fs = std::filesystem;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::milliseconds;
// evo_traj tum groundtruth.txt estimated.txt -p --plot_mode xyz

// TODO:
// optimizare map -> local map -> pentru a creste viteza de procesare 
// reprezentare grafica in spatiu 3d -> folosind pangolin
// separare si abstractizare pentru a putea testa mai multe implementari ale diversilor algoritmi de matching intre frame - uri
// finalizare functie keyframe culling in local mapping
// testare pe alt set de date
// abstractizarea constante importante 

TumDatasetReader *reader;
Tracker *tracker;
int total_tracking_duration = 0;
int total_local_mapping_duration = 0;

void display_timing_information() {  
    std::cout << tracker->reference_kf->reference_idx << " atatea keyframe-uri avute\n";
    std::cout << total_local_mapping_duration / 1000 << " atat a durat doar local mapping\n\n";

    std::cout << total_tracking_duration / 1000 << " atat a durat tracking-ul in total\n";
    std::cout << tracker->orb_matching_time / 1000 << " orb feature matching time\n";
    std::cout << tracker->motion_only_ba_time / 1000 << " motion only BA time\n";
}

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    reader->write_all_entries();
    display_timing_information();
    
    reader->outfile.close();
    exit(signum);
}

// documentatie + prezentare + complexitate + medie generala


// functionat implementarea de pe git
// de verificat printr-o scena 3d ->
// de testat fiecare componenta cu mai multe metode -> FLANN, Brute Force,

// try to use perspective npoint for matching
// care varianta e cea mai buna
// RGBD -> time of flight, structured light

int main()
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
    
    Map *mapp = new Map();
    reader = new TumDatasetReader(cfg, mapp); 
    LocalMapping *local_mapper = new LocalMapping(mapp);
    tracker = new Tracker(mapp, cfg, voc);

    auto t1 = high_resolution_clock::now();
    while(!reader->should_end()) {
        cv::Mat frame = reader->get_next_frame(); 
        cv::Mat depth = reader->get_next_depth();   
        Sophus::SE3d groundtruth_pose = reader->get_next_groundtruth_pose();
        
        auto start = high_resolution_clock::now();
        KeyFrame *kf = tracker->tracking(frame, depth, groundtruth_pose);
        auto end = high_resolution_clock::now();
        total_tracking_duration += duration_cast<milliseconds>(end - start).count();
        if (kf->isKeyFrame) {
            auto start = high_resolution_clock::now();
            local_mapper->local_map(kf);
            auto end = high_resolution_clock::now();
            total_local_mapping_duration += duration_cast<milliseconds>(end - start).count();
        }
        reader->store_entry(kf);
        reader->increase_idx();
    }

    reader->write_all_entries();
    auto t2 = high_resolution_clock::now();
    std::cout << duration_cast<seconds>(t2 - t1).count() << "s aici atata a durat\n\n";  
    display_timing_information();
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
