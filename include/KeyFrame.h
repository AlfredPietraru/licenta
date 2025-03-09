#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include "utils.h"
#include <sophus/se3.hpp>

class KeyFrame
{
public:
    int idx;
    typedef std::shared_ptr<KeyFrame> Ptr;
    Sophus::SE3d Tiw;
    Eigen::Matrix3d K;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat orb_descriptors;
    cv::Mat depth_matrix;

    KeyFrame();
    KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<cv::KeyPoint> keypoints,
             cv::Mat orb_descriptors, cv::Mat depth_matrix, int idx);
    bool operator ==(const KeyFrame& lhs);
    Eigen::Vector3d compute_camera_center();
    Eigen::Vector3d fromWorldToImage(Eigen::Vector4d& wcoord);
    Eigen::Vector4d fromImageToWorld(int kp_idx);
};

namespace std
{
    template <>
    struct hash<KeyFrame>
    {
        size_t operator()(const KeyFrame &p) const
        {
            size_t mem = (size_t)&p.Tiw;
            size_t smem = (size_t)&p.K;
            size_t dmem = (size_t)&p.orb_descriptors;
            return mem ^ smem + mem % smem;
        }
    };
}


#endif