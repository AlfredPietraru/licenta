#ifndef DATASET_READER_H
#define DATASET_READER_H
#include <iostream>
#include <vector>
#include <Eigen/Core>
#include <sophus/common.hpp>
#include <cstdio>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <sophus/se3.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include "Map.h"
#include <unordered_set>
namespace fs = std::filesystem;

struct Position_Entry
{
    double timestamp;
    double tx, ty, tz;
    double qx, qy, qz, qw;
    Sophus::SE3d pose;
};


struct FramePoseEntry {
    int last_keyframe_idx;
    Sophus::SE3d pose;
};

class TumDatasetReader
{
public:
    std::vector<std::string> rgb_path;
    std::vector<std::string> depth_path;
    std::string path_to_write;
    std::ofstream outfile;
    std::vector<Sophus::SE3d> groundtruth_poses;
    std::vector<FramePoseEntry> frame_poses;
    cv::dnn::Net net;
    int idx = 0;
    TumDatasetReader(Map *mapp, std::string path_to_write, std::string rgb_paths_location, std::string depth_path_location, 
        std::string mapping_rgb_depth_filename, std::string mapping_rgb_groundtruth);
    Map *mapp;
    cv::Mat get_next_frame();
    cv::Mat get_next_depth();
    cv::Mat get_next_depth_neural_network();
    Sophus::SE3d get_next_groundtruth_pose();
    void increase_idx();
    void store_entry(KeyFrame *current_kf);
    void write_entry(Sophus::SE3d &translation_pose, Sophus::SE3d &pose, int index);
    void write_all_entries(Sophus::SE3d translation_pose);
    bool should_end();
};

#endif