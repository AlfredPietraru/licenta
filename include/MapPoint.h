#ifndef MAP_POINT_H
#define MAP_POINT_H
#include "iostream"
#include "vector"
#include "opencv2/core.hpp"
#include "Eigen/Core"
#include "unordered_set"

class KeyFrame;

class MapPoint {
public:
        // homogenous coordinates of world space
        Eigen::Vector4d wcoord;
        Eigen::Vector3d view_direction;
        cv::Mat orb_descriptor;
        double dmax, dmin;
        std::unordered_map<KeyFrame*, int> belongs_to_keyframes;
        
        MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord,
                cv::Mat orb_descriptor, int idx, float depth);
        void add_reference_kf(KeyFrame *kf, int idx);
        Eigen::Vector3d get_3d_vector();
};

#endif