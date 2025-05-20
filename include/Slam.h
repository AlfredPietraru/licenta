#ifndef SLAM_WHOLE_H
#define SLAM_WHOLE_H

#include "Tracker.h"
#include "LocalMapping.h"
#include "TumDatasetReader.h"
#include "ORBVocabulary.h"

class SLAM {
public:
    SLAM(std::string voc_file, std::string config_file);

    ORBVocabulary *voc = nullptr;
    Map *mapp = nullptr;
    TumDatasetReader *reader = nullptr;
    Tracker *tracker = nullptr;
    MapDrawer *drawer = nullptr;
    LocalMapping *local_mapper = nullptr;

    int total_duration = 0;
    int total_tracking_duration = 0;
    int total_local_mapping_duration = 0;

    void run_slam_systems();
    void display_timing_information();
};


#endif