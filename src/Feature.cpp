#include "../include/Feature.h"

int inline::Feature::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}

Feature::Feature(cv::KeyPoint kp, cv::Mat descriptor, MapPoint *mp, int idx, double stereo_depth) : kp(kp), 
    mp(mp), idx(idx), descriptor(descriptor), stereo_depth(stereo_depth) {}

Feature::Feature(cv::KeyPoint kp, cv::Mat descriptor, int idx, double stereo_depth) : kp(kp), mp(nullptr), 
    idx(idx), descriptor(descriptor), stereo_depth(stereo_depth) {}

void Feature::set_map_point(MapPoint *mp) {
    if (mp == nullptr) {
        this->mp = mp;
        this->current_hamming_distance = ComputeHammingDistance(mp->orb_descriptor, descriptor);    
    } else {
        int new_hamming_distance = ComputeHammingDistance(mp->orb_descriptor, descriptor);
        if (new_hamming_distance < current_hamming_distance) {
            this->mp = mp;
            this->current_hamming_distance = new_hamming_distance; 
        }
    }
}

MapPoint* Feature::get_map_point() {
    return this->mp;
}

cv::KeyPoint Feature::get_key_point() {
    return this->kp;
}
