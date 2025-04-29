#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H
#include "Map.h"
#include "OrbMatcher.h"
#include "BundleAdjustment.h"

class LocalMapping {
public:
    Map *mapp;
    bool first_kf = true;
    std::unordered_set<MapPoint*> recently_added;
    BundleAdjustment *bundleAdjustment;
    LocalMapping(Map *mapp) {
        this->mapp = mapp;
        this->bundleAdjustment = new BundleAdjustment();
    }

    void local_map(KeyFrame *kf);
    void map_points_culling(KeyFrame *curr_kf);
    void delete_map_point(MapPoint *mp);
    int compute_map_points(KeyFrame *kf);
    Eigen::Matrix3d compute_fundamental_matrix(KeyFrame *curr_kf, KeyFrame *neighbour_kf);
    void search_in_neighbours(KeyFrame *kf);
    void KeyFrameCulling(KeyFrame *kf);
};

#endif
