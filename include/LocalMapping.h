#ifndef LOCAL_MAPPING_H
#define LOCAL_MAPPING_H
#include "Map.h"

class LocalMapping {
public:
    Map *mapp;
    LocalMapping(Map *mapp) {
        this->mapp = mapp;
    }

    void local_map(KeyFrame *kf);
    void update_local_map(KeyFrame *reference_kf);
    void map_points_culling(KeyFrame *kf);
    void delete_map_point(MapPoint *mp);
    int compute_map_points(KeyFrame *kf);
    Eigen::Matrix3d compute_fundamental_matrix(KeyFrame *curr_kf, KeyFrame *neighbour_kf);
    void search_in_neighbours(KeyFrame *kf);
};

#endif
