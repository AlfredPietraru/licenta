#ifndef GRAPH_STRUCTURE_H
#define GRAPH_STRUCTURE_H

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <unordered_set>
#include <iostream>
#include "structures.h"
#include "utils.h"

struct GraphEdge
{
    KeyFrame kf;
    int weight;
    GraphEdge(KeyFrame kf, int weight);
    GraphEdge();

    bool operator==(const GraphEdge &other) const;
};

struct GraphEdgeHash
{
    size_t operator()(const GraphEdge &x) const
    {
        size_t res = (size_t)&x;
        return res * (x.weight % 31 + 17);
    }
};

class Graph
{
public:
    KeyFrame kf;
    std::unordered_set<GraphEdge, GraphEdgeHash> edges;

    Graph();
    Graph(KeyFrame kf);
    void add_edge(Graph node, int weight);
};

#endif