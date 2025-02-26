#ifndef LOCAL_MAP_H
#define LOCAL_MAP_H
#include "MapPoint.h"
#include "KeyFrame.h"
#include "Graph.h"
#include "Map.h"


class LocalMap {
public:
    int graph_nodes_nr = 0;
    Graph graph;
    Graph spanningTree;
    Map mapp;
    
    LocalMap(Graph &graph, Map &map);
    void Add_New_KeyFrame_Graph(std::vector<KeyFrame*> &keyframes_buffer);
};


#endif