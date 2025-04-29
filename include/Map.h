#ifndef MAP_STRUCTURE_H
#define MAP_STRUCTURE_H
#include <unordered_set>
#include <set>
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


class Map {
public:
    std::vector<KeyFrame*> keyframes;
    std::unordered_map<KeyFrame*, std::unordered_map<KeyFrame*, int>> graph;
    std::unordered_set<MapPoint*> local_map;

    Map() {}
    std::unordered_set<KeyFrame*> get_local_keyframes(KeyFrame *kf);
    void track_local_map(KeyFrame *curr_kf, KeyFrame *ref, std::unordered_set<KeyFrame*>& keyframes_already_found);
    std::unordered_set<MapPoint*> get_all_map_points();
    void add_first_keyframe(KeyFrame *new_kf);
    std::unordered_set<MapPoint*> add_new_keyframe(KeyFrame *kf);

    
    std::unordered_map<KeyFrame*, int> get_keyframes_connected(KeyFrame *kf, int limit);
    void update_local_map(KeyFrame *ref, std::unordered_set<KeyFrame*>& keyframes_already_found);
    static bool remove_keyframe_reference_from_map_point(MapPoint *mp, KeyFrame *kf);  
    static bool add_keyframe_reference_to_map_point(MapPoint *mp, Feature *f, KeyFrame *kf);
    static bool add_map_point_to_keyframe(KeyFrame *kf, Feature *f, MapPoint *mp); 
    static bool remove_map_point_from_keyframe(KeyFrame *kf, MapPoint *mp);
    static bool replace_map_points_in_keyframe(KeyFrame *kf, MapPoint *old_mp, MapPoint *new_mp);
    bool update_graph_connections(KeyFrame *kf1, KeyFrame *kf2);
    int get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2);
    void debug_map(KeyFrame *kf);
    static int check_valid_features_number(KeyFrame *kf);
};

#endif
