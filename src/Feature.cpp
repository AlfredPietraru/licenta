#include "../include/Feature.h"

int Feature::ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

Feature::Feature(cv::KeyPoint kp, cv::KeyPoint kpu, cv::Mat descriptor, int idx, double depth, double stereo_depth) : kp(kp), mp(nullptr), 
    idx(idx), descriptor(descriptor), stereo_depth(stereo_depth), kpu(kpu), depth(depth) {}

bool Feature::set_map_point(MapPoint *mp) {
    if (this->mp == nullptr) {
        this->mp = mp;
        this->curr_hamming_dist = ComputeHammingDistance(mp->orb_descriptor, this->descriptor);
        return true;
    } 
    int new_hamming_dist = ComputeHammingDistance(mp->orb_descriptor, this->descriptor);
    if (new_hamming_dist >= curr_hamming_dist) return false;
    this->mp = mp;
    this->curr_hamming_dist = new_hamming_dist; 
    return true;
}

void Feature::unmatch_map_point() {
    this->curr_hamming_dist = 0;
    this->mp = nullptr;
}

MapPoint* Feature::get_map_point() {
    return this->mp;
}

cv::KeyPoint Feature::get_key_point() {
    return this->kp;
}

cv::KeyPoint Feature::get_undistorted_keypoint() {
    return this->kpu;
}
