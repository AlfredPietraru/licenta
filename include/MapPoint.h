#ifndef MAP_POINT_H
#define MAP_POINT_H

#include "utils.h"
#include "KeyFrame.h"
#include "unordered_set"

class MapPoint {
public:
        // homogenous coordinates of world space
        Eigen::Vector4d wcoord;
        // Eigen::Vector3d view_direction;
        cv::Mat orb_descriptor;
        double dmax, dmin;
        std::unordered_map<KeyFrame*, int> belongs_to_keyframes;

    MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, double depth, Eigen::Matrix4d camera_pose, 
    Eigen::Vector3d camera_center, cv::Mat orb_descriptor);
    bool operator ==(const MapPoint& lhs);
    bool map_point_belongs_to_keyframe(KeyFrame *kf);
};

namespace std 
{
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