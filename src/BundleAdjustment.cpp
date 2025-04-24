#include "../include/BundleAdjustment.h"


Sophus::SE3d BundleAdjustment::solve_ceres(Map *mapp, KeyFrame *frame) {
   std::unordered_set<KeyFrame*> local_keyframes = mapp->get_local_keyframes(frame);
    if (local_keyframes.size() == 0) {
        std::cout << "CEVA E GRESIT IN BUNDLE ADJUSTMENT ACEST KEYFRAME ESTE IZOLAT\n";
        return frame->Tcw;
    }
    std::unordered_set<MapPoint*> local_map_points;
    for (KeyFrame *kf : local_keyframes) {
        local_map_points.insert(kf->map_points.begin(), kf->map_points.end());
    }
    if (local_map_points.size() == 0) {
        std::cout << "CEVA NU E BINE NU EXISTA DELOC PUNCTE IN LOCAL MAP PENTRU OPTIMIZARE\n";
    }
    std::unordered_set<KeyFrame*> fixed_keyframes;
    for (MapPoint *mp : local_map_points) {
        for (KeyFrame *kf_which_sees_map_point : mp->keyframes) {
            if (local_keyframes.find(kf_which_sees_map_point) == local_keyframes.end()) fixed_keyframes.insert(kf_which_sees_map_point);
        } 
    }
    
    
}