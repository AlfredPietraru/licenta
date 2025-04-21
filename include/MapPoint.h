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
        std::unordered_set<KeyFrame*> keyframes;
        std::vector<cv::Mat> descriptor_vector;
        int first_seen_frame_idx;
        int number_times_seen = 0;
        int number_associations = 0;
        bool bad = false;
        
        MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, cv::Mat orb_descriptor);
        cv::Mat compute_distinctive_descriptor();
        int predict_image_scale(double distance);
        bool map_point_should_be_deleted();
        void increase_how_many_times_seen();
        void increase_number_associations();
        void decrease_number_associations();
        void add_observation_map_point(KeyFrame *kf, cv::Mat descriptor, Eigen::Vector3d camera_center_kf);
        Eigen::Vector3d compute_view_direction(Eigen::Vector3d camera_center);
        int ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b);
};

#endif