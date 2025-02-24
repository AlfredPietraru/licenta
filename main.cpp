#include "DBoW3/DBoW3.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include "Tracker/Tracker.h"

using namespace cv;
using namespace std;


double data_current[9] = {520.9, 0.0, 325.1, 0.0, 521.0, 249.7, 0.0, 0.0, 1.0};
Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor> > Intermediate_Map(data_current);
Eigen::Matrix3d K = (Eigen::Matrix3d)Intermediate_Map;

void exit_program_failure(std::string message) {
    cout << "\n" << message << "\n";
    exit(-1);
}

int main(int argc, char **argv)
{
    Tracker *tracker = new Tracker();
    std::pair<Graph, vector<MapPoint>> essential_graph_map_points = tracker->initialize(K);
    Graph essential_graph = essential_graph_map_points.first;
    vector<MapPoint> map_points = essential_graph_map_points.second;
    while(1) {
        tracker->tracking(K);
        return 0;   
    }
}