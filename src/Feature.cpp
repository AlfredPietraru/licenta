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

void Feature::set_map_point(MapPoint *mp) {
    this->mp = mp;
}

MapPoint* Feature::get_map_point() {
    return this->mp;
}

cv::KeyPoint Feature::get_key_point() {
    return this->kp;
}
