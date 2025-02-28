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

    Map();

    std::vector<MapPoint*> get_map_points(KeyFrame *frame);
    int check_common_map_points(KeyFrame *kf1, KeyFrame *kf2); 
    void add_map_points(KeyFrame *frame);
};

#endif
