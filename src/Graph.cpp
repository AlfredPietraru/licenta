#include "../include/Graph.h"

void Graph::add_node_to_graph(Graph *node, int weight) {
    this->edges.insert(std::pair<Graph *, int>(node, weight));
    node->edges.insert(std::pair<Graph *, int>(this, weight));
}

std::vector<Graph*> Graph::dfs_get_all_nodes() {
    std::unordered_set<Graph*> all_nodes;
    std::vector<Graph*> stack;
    all_nodes.insert(this);
    stack.push_back(this);
    while(!stack.empty()) {
        Graph *current = stack.back();
        stack.pop_back();
        for (std::pair<Graph*, int> edge : current->edges) {
            if (all_nodes.find(edge.first) != all_nodes.end()) {
                all_nodes.insert(edge.first);
                stack.push_back(edge.first);
            }
        }
    }
    return std::vector<Graph*>(all_nodes.begin(), all_nodes.end());
}

Graph* Graph::find_keyframe_in_graph(KeyFrame *kf) {
    std::unordered_set<Graph*> all_nodes;
    std::vector<Graph*> stack;
    all_nodes.insert(this);
    stack.push_back(this);
    while(!stack.empty()) {
        Graph *current = stack.back();
        if (current->kf == kf) return current;
        stack.pop_back();
        for (std::pair<Graph*, int> edge : current->edges) {
            if (all_nodes.find(edge.first) == all_nodes.end()) {
                if (edge.first->kf == kf) return edge.first;
                all_nodes.insert(edge.first);
                stack.push_back(edge.first);
            }
        }
    }
    return nullptr;
}


void Graph::Add_New_KeyFrame_SpanningTree(KeyFrame *from_graph_kf, KeyFrame *want_to_join_graph_kf, int weight) {
    Graph *node =this->find_keyframe_in_graph(from_graph_kf);
    if (node == nullptr) return;
    node->add_node_to_graph(new Graph(want_to_join_graph_kf), weight);
}