#include "include/Tracker.h"
// #include "DBoW3/DBoW3.h"
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    // DBoW3::Vocabulary vocab;
    // vocab.load("../vocab_larger.yml.gz"); 
    char x;
    Tracker *tracker = new Tracker();
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
    Map mapp = tracker->initialize(resized, depth);
    vector<KeyFrame*> keyframes_buffer;
    while(1) {
        cap.read(frame);
        cv::resize(frame, resized, cv::Size(224, 224));
        blob = cv::dnn::blobFromImage(resized, 1, Size(224, 224), Scalar(), true, false);
        net.setInput(blob);
        depth = net.forward().reshape(1, 224);
        depth = depth * 10000;
        tracker->tracking(resized, depth, mapp, keyframes_buffer);
    }
    return 0;
}