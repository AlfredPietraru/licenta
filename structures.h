#ifndef STRUCTURES_H
#define STRUCTURES_H
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <unordered_set>

const double FOCAL_LENGTH = 500.0;
const double X_CAMERA_OFFSET = 325.1;
const double Y_CAMERA_OFFSET = 250.7;

class MapPoint {
public:
        Eigen::Vector3d wcoord;
        Eigen::Vector3d view_direction;
        cv::Mat orb_descriptor;
        double dmax, dmin;

    MapPoint(cv::KeyPoint kp, double depth, Eigen::Matrix4d camera_pose, 
    Eigen::Vector3d camera_center, cv::Mat orb_descriptor) {

        float new_x = (kp.pt.x - X_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
        float new_y = (kp.pt.y - Y_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
        Eigen::Vector4d word_coordinates = camera_pose * Eigen::Vector4d(new_x, new_y, depth, 1);
        this->wcoord = Eigen::Vector3d(word_coordinates(0), word_coordinates(1), word_coordinates(2));
        this->view_direction = (this->wcoord - camera_center).normalized();
        this->orb_descriptor = orb_descriptor;
        this->dmax = depth * 1.2; // inca nicio idee de ce 
        this->dmin = depth * 0.8;  // inca nicio idee de ce 

    }
};

class KeyFrame {
public:
    Eigen::Matrix4d Tiw;
    Eigen::Matrix3d intrisics;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat orb_descriptors;
    cv::Mat depth_matrix;

    KeyFrame() {};

    KeyFrame(Eigen::Matrix4d Tiw, Eigen::Matrix3d intrisics, std::vector<cv::KeyPoint> keypoints, 
            cv::Mat orb_descriptors, cv::Mat depth_matrix) 
    : Tiw(Tiw), intrisics(intrisics), orb_descriptors(orb_descriptors), keypoints(keypoints), depth_matrix(depth_matrix) {}

};


struct GraphEdge {
    KeyFrame kf;
    int weight;
    GraphEdge(KeyFrame kf, int weight)  {
        this->kf = kf;
        this->weight = weight;
    }
    GraphEdge() {}

    bool operator==(const GraphEdge &other) const{
        return (&kf == &other.kf) && (weight == other.weight);
    }
};

struct GraphEdgeHash {
    size_t operator()(const GraphEdge& x) const {
        size_t res = (size_t)&x;
        return res * (x.weight % 31 + 17);
    }
};

class Graph {
public:
    KeyFrame kf;
    std::unordered_set<GraphEdge, GraphEdgeHash> edges;
    Graph() {}

    Graph(KeyFrame kf) : kf(kf) {
        this->edges = {};
    }
    
    void add_edge(Graph node, int weight) {
        edges.insert(GraphEdge(node.kf, weight));
    }
};

#endif
