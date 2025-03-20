#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include <unordered_set>
#include "Feature.h"


class MapPoint;

class KeyFrame
{
public:
    int idx;
    Sophus::SE3d Tiw;
    Eigen::Matrix3d K;
    std::vector<Feature> features;
    std::unordered_set<MapPoint*> map_points;
    int nr_map_points = 0;
    cv::Mat grid;
    cv::Mat orb_descriptors;
    cv::Mat frame;
    cv::Mat depth_matrix;

    KeyFrame();
    KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<cv::KeyPoint> keypoints,
             cv::Mat orb_descriptors, cv::Mat depth_matrix, int idx, cv::Mat frame);
    Eigen::Vector3d compute_camera_center();
    Eigen::Vector3d fromWorldToImage(Eigen::Vector4d& wcoord);
    Eigen::Vector4d fromImageToWorld(int kp_idx);
    float compute_depth_in_keypoint(cv::KeyPoint kp);
    std::vector<cv::KeyPoint> get_all_keypoints(); 
    Eigen::Vector3d get_viewing_direction();
    void correlate_map_points_to_features_current_frame(std::unordered_map<MapPoint *, Feature*>& matches);
    std::vector<MapPoint *> return_map_points();
    cv::KeyPoint get_keypoint(int idx);
    std::vector<int> get_vector_keypoints_after_reprojection(double u, double v, int window); 
};

#endif