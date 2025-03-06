#ifndef MAP_STRUCTURE_H
#define MAP_STRUCTURE_H
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include "MapPoint.h"
#include "KeyFrame.h"

class Map {
public:
    int KEYFRAMES_WINDOW = 5; 

    std::vector<std::vector<MapPoint*>> map_points;
    std::vector<std::pair<KeyFrame*, std::unordered_map<KeyFrame*, int>>> graph;

    Map();
    Map(KeyFrame *first_kf);
    
    std::vector<MapPoint*> track_local_map(KeyFrame *curr_kf);

    KeyFrame *get_reference_keyframe(KeyFrame *kf);
    std::vector<MapPoint*> compute_local_map(KeyFrame *kf);
    std::vector<MapPoint*> compute_map_points(KeyFrame *kf);
    std::vector<MapPoint*> get_reprojected_map_points(KeyFrame *frame, KeyFrame *reference_kf);
};

#endif
