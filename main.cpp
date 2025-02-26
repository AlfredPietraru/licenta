#include "include/Tracker.h"
#include "include/LocalMapping.h"


int main(int argc, char **argv)
{
    Tracker *tracker = new Tracker();
    std::pair<Graph, Map> essential_graph_map_points = tracker->initialize();
    Graph essential_graph = essential_graph_map_points.first;
    Map mapp = essential_graph_map_points.second;
    vector<KeyFrame*> keyframes_buffer;
    LocalMap localMap = LocalMap(essential_graph, mapp);
    while(1) {
        tracker->tracking(mapp, keyframes_buffer);
        return 1;
    }
}