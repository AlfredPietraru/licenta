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
    int current_idx;
    ORBVocabulary *voc;
    DBoW2::BowVector bow_vec;
    DBoW2::FeatureVector features_vec;
    KeyFrame *reference_kf;
    int reference_idx;
    int isKeyFrame = false;
    const double POW_OCTAVE[8] = {1, 1.2, 1.44, 1.728, 2.0736, 2.48832, 2.985984, 3.5831808};

    std::vector<Feature> features;
    std::unordered_map<MapPoint*, Feature*> mp_correlations;

    const int GRID_HEIGHT = 48;
    const int GRID_WIDTH = 64;
    int currently_matched_points = 0;
    std::vector<std::vector<std::vector<int>>> grid;
    const double BASELINE = 0.08;
    double minX, maxX, minY, maxY;

    Eigen::Matrix4d mat_camera_world;
    Eigen::Matrix4d mat_world_camera;
    Eigen::Vector3d camera_center_world;
    double pose_vector[7] = {1, 0, 0, 0, 0, 0, 0};
    
    KeyFrame();
    KeyFrame(Sophus::SE3d Tcw, Eigen::Matrix3d K, std::vector<double> distorsion, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& undistored_kps,
             cv::Mat orb_descriptors, cv::Mat depth_matrix, int current_idx, cv::Mat& frame, ORBVocabulary *voc);

    KeyFrame(KeyFrame* old_keyframe, std::vector<cv::KeyPoint> &keypoints, std::vector<cv::KeyPoint> &undistored_kps,
         cv::Mat orb_descriptors, cv::Mat depth_matrix);

    void create_grid_matrix();
    void create_feature_vector(std::vector<cv::KeyPoint> &keypoints, std::vector<cv::KeyPoint> &undistored_kps,
            cv::Mat orb_descriptors, cv::Mat depth_matrix);           
    Eigen::Vector4d fromImageToWorld(int kp_idx);
    Sophus::SE3d compute_pose();
    std::vector<cv::KeyPoint> get_all_keypoints(); 
    std::vector<int> get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave); 
    void set_reference_keyframe(KeyFrame *ref);
    bool check_map_point_in_keyframe(MapPoint *mp);
    bool check_number_close_points();
    void compute_bow_representation();
    void set_keyframe_position(Sophus::SE3d Tcw_new);
    void debug_keyframe(cv::Mat frame, int miliseconds);
    int get_map_points_seen_from_multiple_frames(int nr_frames);
    std::vector<MapPoint*> get_map_points(); 

    bool debug_keyframe_valid();
};

#endif