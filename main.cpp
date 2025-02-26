#include "Tracker/Tracker.h"


int main(int argc, char **argv)
{
    Tracker *tracker = new Tracker();
    std::pair<Graph, Map> essential_graph_map_points = tracker->initialize();
    Graph essential_graph = essential_graph_map_points.first;
    Map mapp = essential_graph_map_points.second;
    while(1) {
        tracker->tracking(mapp);
        return 1;
    }
}