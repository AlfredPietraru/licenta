#include "../include/LocalMapping.h"


void LocalMapping::local_map(KeyFrame *kf) {
    mapp->add_new_keyframe(kf);
    mapp->compute_map_points(kf);
}