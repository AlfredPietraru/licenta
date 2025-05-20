#ifndef SLAM_WHOLE_H
#define SLAM_WHOLE_H

#include "Tracker.h"
#include "LocalMapping.h"
#include "TumDatasetReader.h"
#include "ORBVocabulary.h"


struct Tum_Dataset_Structure {
    std::string path_to_write;
    std::string rgb_paths_location;
    std::string depth_path_location;
    std::string mapping_rgb_depth_filename;
    std::string mapping_rgb_groundtruth;
    Sophus::SE3d translation_pose;
};

class SLAM {
public:
    SLAM(std::string voc_file, std::string config_file, Tum_Dataset_Structure dataset_structure);

    ORBVocabulary *voc = nullptr;
    Map *mapp = nullptr;
    TumDatasetReader *reader = nullptr;
    Tracker *tracker = nullptr;
    MapDrawer *drawer = nullptr;
    LocalMapping *local_mapper = nullptr;

    int total_duration = 0;
    int total_tracking_duration = 0;
    int total_local_mapping_duration = 0;
    Tum_Dataset_Structure dt;

    void run_slam_systems();
    void display_timing_information();
};


#endif