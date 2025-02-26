#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "utils.h"
#include "MapPoint.h"

class KeyFrame
{
public:
    typedef std::shared_ptr<KeyFrame> Ptr;
    Eigen::Matrix4d Tiw;
    Eigen::Matrix3d intrisics;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat orb_descriptors;
    cv::Mat depth_matrix;

    KeyFrame();
    KeyFrame(Eigen::Matrix4d Tiw, Eigen::Matrix3d intrisics, std::vector<cv::KeyPoint> keypoints,
             cv::Mat orb_descriptors, cv::Mat depth_matrix);
    bool operator ==(const KeyFrame& lhs);
    bool map_point_belongs_to_keyframe(MapPoint *mp);
};

namespace std
{
    template <>
    struct hash<KeyFrame>
    {
        size_t operator()(const KeyFrame &p) const
        {
            size_t mem = (size_t)&p.Tiw;
            size_t smem = (size_t)&p.intrisics;
            size_t dmem = (size_t)&p.orb_descriptors;
            return mem ^ smem + mem % smem;
        }
    };
}


#endif