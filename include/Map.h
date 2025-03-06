#ifndef MAP_STRUCTURE_H
#define MAP_STRUCTURE_H
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include "MapPoint.h"
#include "KeyFrame.h"

class Map {
public:
    std::unordered_map<KeyFrame*, std::vector<MapPoint*>> map_points;
    std::unordered_map<KeyFrame*, std::unordered_map<KeyFrame*, int>> graph;
    std::unordered_map<KeyFrame*, std::unordered_map<KeyFrame*, int>> spanning_tree;

    Map();

    Map(KeyFrame *first_kf);

    std::vector<MapPoint*> compute_map_points(KeyFrame *kf);
    std::vector<MapPoint*> get_reprojected_map_points(KeyFrame *frame, KeyFrame *reference_kf);
    std::vector<MapPoint*> track_local_map(KeyFrame *curr_kf, KeyFrame *reference_kf);
    int check_common_map_points(KeyFrame *kf1, KeyFrame *kf2); 
};

#endif
