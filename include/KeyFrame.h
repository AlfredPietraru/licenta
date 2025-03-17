#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include "Feature.h"


class KeyFrame
{
public:
    int idx;
    typedef std::shared_ptr<KeyFrame> Ptr;
    Sophus::SE3d Tiw;
    Eigen::Matrix3d K;
    std::vector<Feature> features;
    // std::vector<cv::KeyPoint> keypoints;
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
};

#endif