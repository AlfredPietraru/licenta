#ifndef MAP_STRUCTURE_H
#define MAP_STRUCTURE_H
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include "MapPoint.h"
#include "KeyFrame.h"
#include "config.h"
#include "OrbMatcher.h"


class Map {
public:
    int KEYFRAMES_WINDOW = 5;
    OrbMatcher* matcher;
    std::vector<KeyFrame*> keyframes;
    std::unordered_map<KeyFrame*, std::unordered_map<KeyFrame*, int>> graph;

    Map();
    Map(Orb_Matcher orb_matcher_cfg);
    std::unordered_map<MapPoint *, Feature*> track_local_map(KeyFrame *curr_kf, KeyFrame *reference_kf);
    KeyFrame *get_reference_keyframe(KeyFrame *kf);
    std::unordered_set<MapPoint *> compute_local_map(KeyFrame *current_frame, KeyFrame *reference_frame);
    std::unordered_set<MapPoint*> get_all_map_points();
    void compute_map_points(KeyFrame *kf);
    void add_first_keyframe(KeyFrame *new_kf);
    void add_new_keyframe(KeyFrame *kf); 
    void debug_map(KeyFrame *kf);
};

#endif
