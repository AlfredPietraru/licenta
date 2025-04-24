#ifndef MAP_POINT_H
#define MAP_POINT_H
#include "iostream"
#include "vector"
#include <iostream>
#include "opencv2/core.hpp"
#include "Eigen/Core"
#include "unordered_set"
#include <set>

class KeyFrame;

class MapPointEntry {
public:
        int feature_idx;
        Eigen::Vector3d camera_center;
        cv::Mat descriptor;
        MapPointEntry(int feature_idx, Eigen::Vector3d camera_center, cv::Mat descriptor) :
        feature_idx(feature_idx), camera_center(camera_center), descriptor(descriptor) {}
};

class MapPoint {
public:
        // homogenous coordinates of world space
        Eigen::Vector4d wcoord;
        Eigen::Vector3d wcoord_3d;
        Eigen::Vector3d view_direction;
        cv::Mat orb_descriptor;
        double dmax, dmin;
        double BASELINE = 0.08;

        std::vector<KeyFrame*> keyframes;
        std::unordered_map<KeyFrame*, MapPointEntry*> data;
        int first_seen_frame_idx;
        int number_times_seen = 0;
        int number_associations = 0;
        bool bad = false;
        int octave = 0;
        
        MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, cv::Mat orb_descriptor);
        void compute_distinctive_descriptor();
        void compute_distance();
        int predict_image_scale(double distance);
        bool map_point_should_be_deleted();
        void increase_how_many_times_seen();
        void increase_number_associations(int val);
        void decrease_number_associations(int val);
        void add_observation(KeyFrame *kf, Eigen::Vector3d camera_center, cv::Mat orb_descriptor);
        void remove_observation(KeyFrame *kf);
        void compute_view_direction();
        int ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b);
        bool find_keyframe(KeyFrame *kf);
};

#endif