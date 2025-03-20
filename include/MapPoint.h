#ifndef MAP_POINT_H
#define MAP_POINT_H

#include "KeyFrame.h"
#include "unordered_set"

class MapPoint {
public:
        // homogenous coordinates of world space
        Eigen::Vector4d wcoord;
        Eigen::Vector3d view_direction;
        cv::Mat orb_descriptor;
        double dmax, dmin;
        std::unordered_map<KeyFrame*, int> belongs_to_keyframes;
        
        MapPoint(KeyFrame *keyframe, int idx, float depth);
        void add_reference_kf(KeyFrame *kf, int idx);
        int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
        int reproject_map_point(KeyFrame *kf, int window, int orb_descriptor_value);
        Eigen::Vector3d get_3d_vector();
};

#endif