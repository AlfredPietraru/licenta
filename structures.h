#ifndef PRIMITIVE_STRUCTURES_H
#define PRIMITIVE_STRUCTURES_H

#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "utils.h"

class KeyFrame
{
public:
    typedef std::shared_ptr<KeyFrame> Ptr;
    Eigen::Matrix4d Tiw;
    Eigen::Matrix3d intrisics;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat orb_descriptors;
    cv::Mat depth_matrix;

    KeyFrame() {};

    KeyFrame(Eigen::Matrix4d Tiw, Eigen::Matrix3d intrisics, std::vector<cv::KeyPoint> keypoints,
             cv::Mat orb_descriptors, cv::Mat depth_matrix)
        : Tiw(Tiw), intrisics(intrisics), orb_descriptors(orb_descriptors), keypoints(keypoints), depth_matrix(depth_matrix) {}

    bool operator ==(const KeyFrame& lhs)
    {
        return (size_t)this == (size_t)&lhs;
    }
};

class MapPoint {
public:
        typedef std::shared_ptr<MapPoint> Ptr;
        // homogenous coordinates of world space
        Eigen::Vector4d wcoord;
        // Eigen::Vector3d view_direction;
        cv::Mat orb_descriptor;
        double dmax, dmin;

    MapPoint(cv::KeyPoint kp, double depth, Eigen::Matrix4d camera_pose, 
    Eigen::Vector3d camera_center, cv::Mat orb_descriptor) {

        float new_x = (kp.pt.x - X_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
        float new_y = (kp.pt.y - Y_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
        this->wcoord = camera_pose * Eigen::Vector4d(new_x, new_y, depth, 1);
        // this->view_direction = (this->wcoord - camera_center).normalized();
        this->orb_descriptor = orb_descriptor;
        this->dmax = depth * 1.2; // inca nicio idee de ce 
        this->dmin = depth * 0.8;  // inca nicio idee de ce 
    }
    
    bool operator ==(const MapPoint& lhs)
    {
        return (size_t)this == (size_t)&lhs;
    }
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

    template <>
    struct hash<MapPoint>
    {
        size_t operator()(const MapPoint p) const 
        {
            size_t mem = (size_t)&p.orb_descriptor;
            size_t smem = (size_t)&p.wcoord;
		    return mem ^ smem + mem % smem;
        }
	};
}

#endif