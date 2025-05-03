#include "../include/Feature.h"

Feature::Feature(cv::KeyPoint kp, cv::KeyPoint kpu, cv::Mat descriptor, int idx, double depth, double right_coordinate) : kp(kp), kpu(kpu),
    descriptor(descriptor), idx(idx), depth(depth), right_coordinate(right_coordinate) {}

void Feature::set_map_point(MapPoint *mp, int hamming_distance) {
    this->mp = mp;
    this->curr_hamming_dist = hamming_distance;
}

void Feature::unmatch_map_point() {
    this->curr_hamming_dist = 10000;
    this->mp = nullptr;
}

MapPoint* Feature::get_map_point() {
    return this->mp;
}

cv::KeyPoint  Feature::get_key_point() {
    return this->kp;
}

cv::KeyPoint Feature::get_undistorted_keypoint() {
    return this->kpu;
}
