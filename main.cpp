#include "Tracker/Tracker.h"


int main(int argc, char **argv)
{
    Tracker *tracker = new Tracker();
    std::pair<Graph, std::vector<MapPoint>> essential_graph_map_points = tracker->initialize();
    Graph essential_graph = essential_graph_map_points.first;
    std::vector<MapPoint> map_points = essential_graph_map_points.second;
    while(1) {
        tracker->tracking(map_points);
        return 1;
    }
}