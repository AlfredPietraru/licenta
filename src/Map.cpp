#include "../include/Map.h"

Map::Map() {}

std::vector<MapPoint*> Map::compute_map_points(KeyFrame *frame) {
    std::vector<MapPoint *> current_points_found;
    // std::cout << frame->keypoints.size() << " keypoint-uri avute \n";
    for (int i = 0; i < frame->keypoints.size(); i++) {
        cv::KeyPoint kp = frame->keypoints[i];
        float dd = frame->compute_depth_in_keypoint(kp);
        if (dd <= 0) continue;
        current_points_found.push_back(new MapPoint(frame, i, dd));
    }
    std::cout << current_points_found.size() <<  " puncte gasite \n";
    if (current_points_found.size() == 0) return {};
    return current_points_found;
}

Map::Map(KeyFrame *first_kf) {
    std::vector<MapPoint*> kf_map_points = this->compute_map_points(first_kf);
    this->map_points.push_back(kf_map_points);
    this->graph.push_back(std::pair<KeyFrame*, std::unordered_map<KeyFrame*, int>>(first_kf, {}));
}

std::vector<MapPoint *> Map::get_reprojected_map_points(KeyFrame *curr_frame, KeyFrame *reference_kf)
{
    std::vector<MapPoint*> reference_map_points = this->map_points[reference_kf->idx];
    // std::cout << "\n\n" << reference_map_points.size() << "\n\n";
    if ( reference_map_points.size() == 0) return {};
    std::vector<MapPoint*> out;
    for (MapPoint *mp : reference_map_points)
    {
        if (mp->map_point_belongs_to_keyframe(curr_frame)) {
            out.push_back(mp);
        }
    }
    return out;
}


KeyFrame *Map::get_reference_keyframe(KeyFrame *kf) {
    KeyFrame* last_kf = this->graph.back().first;
    int last_idx = last_kf->idx;
    int max = -1;
    int reference_idx = last_idx;
    for (int i = 0; i < this->KEYFRAMES_WINDOW; i++) {
        int current_idx = last_idx - i;
        if (current_idx < 0) break;
        std::vector<MapPoint*> reprojected_map_points = get_reprojected_map_points(kf, this->graph[current_idx].first);
        if (reprojected_map_points.size() > max) {
            reference_idx = current_idx;
            max = reprojected_map_points.size();
        } 
    }
    return this->graph[reference_idx].first;
}

std::vector<MapPoint*> Map::compute_local_map(KeyFrame *kf) {
    KeyFrame *reference_kf = get_reference_keyframe(kf);
    std::unordered_map<KeyFrame*, int> reference_kf_neighbours = this->graph[reference_kf->idx].second;
    std::vector<MapPoint*> out = get_reprojected_map_points(kf, reference_kf);
    for (std::pair<KeyFrame*, int> graph_edge : reference_kf_neighbours) {
        std::vector<MapPoint*> reprojected_map_points = get_reprojected_map_points(kf, graph_edge.first);
        out.insert(out.end(), reprojected_map_points.begin(), reprojected_map_points.end());
    }
    return out;
}


// INCOMPLET, vor exista map point-uri duplicate
std::pair<std::vector<MapPoint*>, std::vector<cv::KeyPoint>> Map::track_local_map(KeyFrame *curr_kf) {
    std::vector<MapPoint*> local_map = this->compute_local_map(curr_kf);
    std::vector<MapPoint*> out_map;
    std::vector<cv::KeyPoint> kps;
    for (MapPoint *mp : local_map) {
        int idx = mp->find_orb_correspondence(curr_kf);
        if (idx == -1) continue;
        out_map.push_back(mp);
        kps.push_back(curr_kf->keypoints[idx]);
    }
    return std::pair<std::vector<MapPoint*>, std::vector<cv::KeyPoint>>(out_map, kps);
}