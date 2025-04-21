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
    int current_idx;
    Sophus::SE3d Tcw;
    Eigen::Matrix3d K;
    std::vector<Feature> features;
    std::unordered_map<MapPoint*, Feature*> mp_correlations; 
    std::unordered_set<MapPoint*> map_points;
    std::unordered_set<MapPoint*> outliers;
    KeyFrame *reference_kf;
    
    int GRID_HEIGHT = 64;
    int GRID_WIDTH = 48;
    int currently_matched_points = 0;
    std::vector<std::vector<std::vector<int>>> grid;
    cv::Mat orb_descriptors;
    const double BASELINE = 0.08;
    float minX, maxX, minY, maxY;
    
    ORBVocabulary *voc;
    DBoW2::BowVector bow_vec;
    DBoW2::FeatureVector features_vec;


    KeyFrame();
    KeyFrame(Sophus::SE3d Tcw, Eigen::Matrix3d K, std::vector<double> distorsion, std::vector<cv::KeyPoint>& keypoints, std::vector<cv::KeyPoint>& undistored_kps,
             cv::Mat orb_descriptors, cv::Mat depth_matrix, int current_idx, cv::Mat& frame, ORBVocabulary *voc, KeyFrame *reference_kf);
    Eigen::Vector3d compute_camera_center_world();
    Eigen::Vector3d fromWorldToImage(Eigen::Vector4d& wcoord);
    Eigen::Vector4d fromImageToWorld(int kp_idx);
    Eigen::Vector3d fromImageToWorld_3d(int kp_idx);
    std::vector<cv::KeyPoint> get_all_keypoints(); 
    std::vector<int> get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave); 
    void add_map_point(MapPoint *mp, Feature *f, cv::Mat orb_descriptor);
    void remove_map_point(MapPoint *mp);
    void add_outlier_element(MapPoint *mp);
    void remove_outlier_element(MapPoint *mp);
    bool check_map_point_outlier(MapPoint *mp);
    bool check_number_close_points();
    void compute_bow_representation();
    void update_map_points_info();
    void set_keyframe_position(Sophus::SE3d Tcw_new);
    void debug_keyframe(cv::Mat frame, int miliseconds, std::unordered_map<MapPoint*, Feature*>& matches,std::unordered_map<MapPoint*, Feature*>& new_matches);

private:
    Eigen::Matrix4d mat_camera_world;
    Eigen::Matrix4d mat_world_camera;
};

#endif