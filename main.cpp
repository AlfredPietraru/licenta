#include "include/Tracker.h"
// #include "DBoW3/DBoW3.h"
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>
#include <filesystem>
namespace fs = std::filesystem;

using namespace std;


void read_img() {
    cv::VideoCapture cap = cv::VideoCapture("../video.mp4");
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../fast_depth.onnx");
    Mat frame;
    cap.read(frame);
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(224, 224));
    Mat blob = cv::dnn::blobFromImage(resized, 1, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat depth = net.forward().reshape(1, 224);
    depth = depth * 10000;
}



void read_from_dataset() {
    std::string rgb_path = "../rgbd_dataset_freiburg1_rpy/rgb";
    std::string depth_path = "../rgbd_dataset_freiburg1_rpy/depth";
    Tracker *tracker = new Tracker();
    bool start = true;
    Map mapp;
    Mat frame, depth; 
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
    // for (int i = 0; i < rgb_file_paths.size(); i++) {
    //     std::cout << rgb_file_paths[i] << " " << depth_file_paths[i] << "\n";
    // }

    for (int i = 0; i < rgb_file_paths.size(); i++) {
        frame = cv::imread(rgb_file_paths[i], cv::IMREAD_COLOR_RGB);
        depth = cv::imread(depth_file_paths[i], cv::IMREAD_UNCHANGED);
        std::cout << rgb_file_paths[i]  << " " << frame.size() << " " << depth.size() << " dimensiune imagini \n";
        if (start) {
            mapp = tracker->initialize(frame, depth);
            start = !start;
            continue;
        }
        tracker->tracking(frame, depth, mapp, keyframes_buffer);
    }
}

int main(int argc, char **argv)
{
    cv::VideoCapture cap = cv::VideoCapture("../video.mp4");
    cv::dnn::Net net = cv::dnn::readNetFromONNX("../fast_depth.onnx");
    Tracker *tracker = new Tracker();
    cv::Mat frame, resized, normalized_frame;
    cap.read(frame);
    // de normalizat inputul si de adus in range-ul 0, 1, CE EROARE ELEMENTARAAAA
    cv::resize(frame, resized, cv::Size(224, 224));
    cv::normalize(resized, normalized_frame, 0, 255, cv::NORM_MINMAX);
    Mat blob = cv::dnn::blobFromImage(normalized_frame, 1, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat depth = net.forward().reshape(1, 224);
    depth *= 10;
    Map mapp = tracker->initialize(resized, depth);
    vector<KeyFrame*> keyframes_buffer;
    while(1) {
        cap.read(frame);
        cv::resize(frame, resized, cv::Size(224, 224));
        cv::normalize(resized, normalized_frame, 0, 255, cv::NORM_MINMAX);
        blob = cv::dnn::blobFromImage(normalized_frame, 1, Size(224, 224), Scalar(), true, false);
        net.setInput(blob);
        depth = net.forward().reshape(1, 224);
        depth *= 10;
        tracker->tracking(resized, depth, mapp, keyframes_buffer);
    }
    return 0;
}