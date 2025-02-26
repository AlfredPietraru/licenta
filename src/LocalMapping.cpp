#include "../include/LocalMapping.h"

LocalMap::LocalMap(Graph &graph, Map &mapp) : graph(graph), mapp(mapp) {
    this->spanningTree = Graph(graph.kf);
}


void LocalMap::Add_New_KeyFrame_Graph(std::vector<KeyFrame*> &keyframes_buffer) {
    if (keyframes_buffer.size() == 0) return;
    KeyFrame *keyframe = keyframes_buffer.back();
    keyframes_buffer.pop_back();
    Graph *future_node = new Graph(keyframe); 
    std::vector<Graph*> nodes = graph.dfs_get_all_nodes();
    int biggest_weight = -1;
    KeyFrame *best_kf = nullptr;
    for (Graph *node : nodes) {
        int weight = mapp.check_common_map_points(node->kf, future_node->kf);
        if (weight > biggest_weight) {
            biggest_weight = weight;
            best_kf = node->kf;
        }
        if (weight < 15) continue; 
        node->add_node_to_graph(future_node, weight);
    }
    spanningTree.Add_New_KeyFrame_SpanningTree(best_kf, future_node->kf, biggest_weight);

    // compute_bag_of_words;
    //triangulate
}