#ifndef MAP_POINT_H
#define MAP_POINT_H
#include "iostream"
#include "vector"
#include <iostream>
#include "opencv2/core.hpp"
#include "Eigen/Core"
#include "unordered_set"
class KeyFrame;

class MapPoint {
public:
        // homogenous coordinates of world space
        Eigen::Vector4d wcoord;
        Eigen::Vector3d wcoord_3d;
        Eigen::Vector3d view_direction;
        cv::Mat orb_descriptor;
        double dmax, dmin;
        double BASELINE = 0.08;
        std::unordered_map<KeyFrame*, int> belongs_to_keyframes;
        bool is_outlier = false;
        int first_seen_frame_idx;
        int how_many_times_seen = 0;
        
        MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord,
                cv::Mat orb_descriptor, int idx);
        void add_reference_kf(KeyFrame *kf, int idx);
        Eigen::Vector3d get_3d_vector();
        int check_index_in_keyframe(KeyFrame *kf);
        int predict_image_scale(double distance);
        void increase_how_many_times_seen();
};

#endif