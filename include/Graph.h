#ifndef GRAPH_STRUCTURE_H
#define GRAPH_STRUCTURE_H
#include <unordered_set>
#include <iostream>
#include "utils.h"
#include "MapPoint.h"
#include "KeyFrame.h"

class Graph
{
    struct GraphEdgeHash
    {
        size_t operator()(const std::pair<Graph *, int> &x) const
        {
            return ((size_t)&x.first) * 31 + x.second;
        }
    };

public:
    KeyFrame *kf;
    std::unordered_set<std::pair<Graph *, int>, GraphEdgeHash> edges;

    Graph() {}
    Graph(KeyFrame *kf)
    {
        this->kf = kf;
        this->edges = {};
    }

    void add_node_to_graph(Graph *node, int weight);
    std::vector<Graph *> dfs_get_all_nodes();
    Graph *find_keyframe_in_graph(KeyFrame *kf);
    void Add_New_KeyFrame_SpanningTree(KeyFrame *from_graph_kf, KeyFrame *want_to_join_graph_kf, int weight);
};

#endif