#ifndef KEYFRAME_H
#define KEYFRAME_H

#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <sophus/se3.hpp>
#include <unordered_set>
#include "ORBVocabulary.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include "Feature.h"
#include "MapPoint.h"

class KeyFrame
{
public:
    Sophus::SE3d Tcw;
    Eigen::Matrix3d K;
    cv::Mat orb_descriptors;
    int current_idx;
    ORBVocabulary *voc;
    DBoW2::BowVector bow_vec;
    DBoW2::FeatureVector features_vec;
    KeyFrame *reference_kf;
    int reference_idx;

    std::vector<Feature> features;
    std::unordered_map<MapPoint*, Feature*> mp_correlations; 
    std::unordered_set<MapPoint*> map_points;
    std::unordered_set<MapPoint*> outliers;

    int GRID_HEIGHT = 64;
    int GRID_WIDTH = 48;
    int currently_matched_points = 0;
    std::vector<std::vector<std::vector<int>>> grid;
    const double BASELINE = 0.08;
    float minX, maxX, minY, maxY;
    
    KeyFrame();
    KeyFrame(Sophus::SE3d Tcw, Eigen::Matrix3d K, std::vector<double> distorsion, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& undistored_kps,
             cv::Mat orb_descriptors, cv::Mat depth_matrix, int current_idx, cv::Mat& frame, ORBVocabulary *voc, KeyFrame *reference_kf, int reference_idx);
    Eigen::Vector3d fromWorldToImage(Eigen::Vector4d& wcoord);
    Eigen::Vector4d fromImageToWorld(int kp_idx);
    std::vector<cv::KeyPoint> get_all_keypoints(); 
    std::vector<int> get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave); 
    void add_outlier_element(MapPoint *mp);
    void remove_outlier_element(MapPoint *mp);
    bool check_map_point_in_keyframe(MapPoint *mp);
    bool check_map_point_outlier(MapPoint *mp);
    bool check_number_close_points();
    void compute_bow_representation();
    void set_keyframe_position(Sophus::SE3d Tcw_new);
    void debug_keyframe(cv::Mat frame, int miliseconds);
    int get_map_points_seen_from_multiple_frames(int nr_frames);
    double *compute_vector_pose();
    
    Eigen::Matrix4d mat_camera_world;
    Eigen::Matrix4d mat_world_camera;
    Eigen::Vector3d camera_center_world;
};

#endif