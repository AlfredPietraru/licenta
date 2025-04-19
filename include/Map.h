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
    std::unordered_set<MapPoint*> local_map;

    Map();
    Map(Orb_Matcher orb_matcher_cfg);
    std::unordered_set<KeyFrame*> get_local_keyframes(KeyFrame *kf);
    void track_local_map(std::unordered_map<MapPoint *, Feature*> &matches, KeyFrame *curr_kf, KeyFrame *reference_kf);
    std::unordered_set<MapPoint*> get_all_map_points();
    void add_first_keyframe(KeyFrame *new_kf);
    void add_new_keyframe(KeyFrame *kf);
    int get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2);
    void debug_map(KeyFrame *kf);
};

#endif
