#include "../include/MapPoint.h"

MapPoint::MapPoint(cv::KeyPoint kp, double depth, Eigen::Matrix4d camera_pose,
                   Eigen::Vector3d camera_center, cv::Mat orb_descriptor)
{

    float new_x = (kp.pt.x - X_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
    float new_y = (kp.pt.y - Y_CAMERA_OFFSET) * depth / FOCAL_LENGTH;
    this->wcoord = camera_pose * Eigen::Vector4d(new_x, new_y, depth, 1);
    // this->view_direction = (this->wcoord - camera_center).normalized();
    this->orb_descriptor = orb_descriptor;
    this->dmax = depth * 1.2; // inca nicio idee de ce
    this->dmin = depth * 0.8; // inca nicio idee de ce
}

bool MapPoint::operator==(const MapPoint &lhs)
{
    return (size_t)this == (size_t)&lhs;
}