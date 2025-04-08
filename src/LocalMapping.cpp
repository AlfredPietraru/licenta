#include "../include/LocalMapping.h"


void LocalMapping::local_map(KeyFrame *kf) {
    if (kf == nullptr) return;
    // add new map points;
    mapp.add_new_keyframe(kf);

    
}