#include "../include/Feature.h"

int inline::Feature::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}

Feature::Feature(cv::KeyPoint kp, cv::Mat descriptor, KeyFrame *frame, MapPoint *mp, int idx) : kp(kp), frame(frame), 
    mp(mp), idx(idx), descriptor(descriptor) {}

Feature::Feature(cv::KeyPoint kp, cv::Mat descriptor, KeyFrame *frame, int idx) : kp(kp), frame(frame), 
    mp(nullptr), idx(idx), descriptor(descriptor) {}

void Feature::set_map_point(MapPoint *mp) {
    if (mp != nullptr) {
        this->mp = mp;    
    } else {
        std::cout << "Map Point-ul asociat este NULL\n";
    }
}
void Feature::set_key_frame(KeyFrame *frame) {
    if (frame != nullptr) {
        this->frame = frame;
    } else {
        std::cout << "FRAME-UL ASOCIAT ESTE NULL\n";
    }
}

MapPoint* Feature::get_map_point() {
    return this->mp;
}

KeyFrame* Feature::get_key_frame() {
    return this->frame;
}

cv::KeyPoint Feature::get_key_point() {
    return this->kp;
}
