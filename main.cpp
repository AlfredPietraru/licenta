#include "include/Tracker.h"
#include "include/LocalMapping.h"
// #include "DBoW3/DBoW3.h"
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>

using namespace std;

int main(int argc, char **argv)
{
    // DBoW3::Vocabulary vocab;
    // vocab.load("../vocab_larger.yml.gz"); 
    char x;
    Tracker *tracker = new Tracker();
    std::pair<Graph, Map> essential_graph_map_points = tracker->initialize();
    Graph essential_graph = essential_graph_map_points.first;
    Map mapp = essential_graph_map_points.second;
    vector<KeyFrame*> keyframes_buffer;
    LocalMap localMap = LocalMap(essential_graph, mapp);
    while(1) {
        tracker->tracking(mapp, keyframes_buffer);
    }
    return 0;
}