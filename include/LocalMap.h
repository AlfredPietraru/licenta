#ifndef LOCAL_MAP_H
#define LOCAL_MAP_H
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Graph.h"
#include "Map.h"


class LocalMap {
public:
    Graph graph;
    Map mapp;
    
    LocalMap(Graph &graph, Map &map);
    void Add_New_KeyFrame(std::vector<KeyFrame*> &keyframes_buffer);


private:
    int check_common_map_points(KeyFrame *first, KeyFrame *second);
};


#endif