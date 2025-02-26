#include "../include/Graph.h"

GraphEdge::GraphEdge() {}
GraphEdge::GraphEdge(KeyFrame *kf, int weight) {
        this->kf = kf;
        this->weight = weight;
}
bool GraphEdge::operator ==(const GraphEdge &other) const
{
    return (&kf == &other.kf) && (weight == other.weight);
}

Graph::Graph() {}
Graph::Graph(KeyFrame *kf) : kf(kf)
{
    this->edges = {};
}

void Graph::add_edge(Graph node, int weight)
{
        edges.insert(GraphEdge(node.kf, weight));
}
