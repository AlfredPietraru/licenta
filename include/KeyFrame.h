#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <unordered_set>
#include "Feature.h"
#include <iostream>
#include "ORBVocabulary.h"

class KeyFrame
{
public:
    int current_idx;
    Sophus::SE3d Tiw;
    Eigen::Matrix3d K;
    std::vector<Feature> features;
    std::unordered_set<MapPoint*> map_points;
    std::unordered_set<MapPoint*> outliers;

    int currently_matched_points = 0;
    cv::Mat grid;
    cv::Mat orb_descriptors;
    cv::Mat frame;
    cv::Mat depth_matrix;
    const double BASELINE = 0.08;

    DBoW2::BowVector bow_vec;
    DBoW2::FeatureVector features_vec;


    KeyFrame();
    KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<double> distorsion, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& undistored_kps,
             cv::Mat orb_descriptors, cv::Mat depth_matrix, int current_idx, cv::Mat& frame, ORBVocabulary *voc);
    Eigen::Vector3d compute_camera_center();
    Eigen::Vector3d fromWorldToImage(Eigen::Vector4d& wcoord);
    Eigen::Vector4d fromImageToWorld(int kp_idx);
    float compute_depth_in_keypoint(cv::KeyPoint kp);
    std::vector<cv::KeyPoint> get_all_keypoints(); 
    Eigen::Vector3d get_viewing_direction();
    void correlate_map_points_to_features_current_frame(std::unordered_map<MapPoint *, Feature*>& matches);
    std::vector<int> get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave); 
    std::unordered_set<MapPoint*> return_map_points_frame();
    std::unordered_map<MapPoint*, Feature*> return_map_points_keypoint_correlation();
    void add_outlier_element(MapPoint *mp);
    void remove_outlier_element(MapPoint *mp);
    bool check_map_point_outlier(MapPoint *mp);
    int check_possible_close_points_generation();
    int check_number_close_points();
    void compute_bow_representation(ORBVocabulary *voc);

    void debug_keyframe(int miliseconds, std::unordered_map<MapPoint*, Feature*>& matches,std::unordered_map<MapPoint*, Feature*>& new_matches);
};

#endif