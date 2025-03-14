#include "../include/Feature.h"

Feature::Feature(cv::KeyPoint kp, KeyFrame *frame, MapPoint *mp) : kp(kp), frame(frame), mp(mp) {}

Feature::Feature(cv::KeyPoint kp, KeyFrame *frame) : kp(kp), frame(frame), mp(nullptr) {}

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
