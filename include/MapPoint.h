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

    float WINDOW = 20; 
    MapPoint(KeyFrame *keyframe, int idx, float depth);
    bool map_point_belongs_to_keyframe(KeyFrame *kf);
    void add_reference_kf(KeyFrame *kf, int idx);
    int find_orb_correspondence(KeyFrame *kf);
    int ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2);
};

#endif