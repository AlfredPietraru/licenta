
// de folosit sophus pentru estimarea pozitiei
#include <cstdio> 
#include <iostream>
#include <filesystem>
#include "include/Slam.h"
#include <csignal>
#include <cstdlib>
#include <chrono>
namespace fs = std::filesystem;
using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::seconds;
using std::chrono::milliseconds;
// evo_traj tum groundtruth.txt estimated.txt -p --plot_mode xyz
// TODO:
// separare si abstractizare pentru a putea testa mai multe implementari ale diversilor algoritmi de matching intre frame - uri, de testat fiecare componenta cu mai multe metode -> FLANN, Brute Force, PnP
// testare pe alt set de date
// abstractizarea constante importante

SLAM *slam = nullptr;

void signalHandler(int signum) {
    std::cout << "Interrupt signal (" << signum << ") received.\n";
    if (slam != nullptr) {
        slam->display_timing_information();
        slam->reader->outfile.close();
    }
    exit(signum);
}

int main()
{
    std::signal(SIGINT, signalHandler);
    Tum_Dataset_Structure dt;
    // dt.path_to_write = "../rgbd_dataset_freiburg1_xyz/estimated.txt";
    // dt.rgb_paths_location = "../rgbd_dataset_freiburg1_xyz/rgb";
    // dt.depth_path_location = "../rgbd_dataset_freiburg1_xyz/depth";
    // dt.mapping_rgb_depth_filename = "../rgbd_dataset_freiburg1_xyz/maping_rgb_depth.txt";
    // dt.mapping_rgb_groundtruth = "../rgbd_dataset_freiburg1_xyz/maping_rgb_groundtruth.txt";
    // dt.translation_pose = Sophus::SE3d(Eigen::Quaterniond(-0.3266, 0.6583, 0.6112, -0.2938), Eigen::Vector3d(1.3434, 0.6271, 1.6606));


    dt.path_to_write = "../rgbd_dataset_freiburg1_rpy/estimated.txt";
    dt.rgb_paths_location = "../rgbd_dataset_freiburg1_rpy/rgb";
    dt.depth_path_location = "../rgbd_dataset_freiburg1_rpy/depth";
    dt.mapping_rgb_depth_filename = "../rgbd_dataset_freiburg1_rpy/mapping_rgb_depth.txt";
    dt.mapping_rgb_groundtruth = "../rgbd_dataset_freiburg1_rpy/mapping_rgb_groundtruth.txt";
    dt.translation_pose = Sophus::SE3d(Eigen::Quaterniond(-0.3496, 0.6504, 0.6012, -0.3056), Eigen::Vector3d(1.3357, 0.6698, 1.6180));

    slam = new SLAM("../ORBvoc.txt", "../config.yaml", dt);
    slam->run_slam_systems();
    slam->display_timing_information();
}