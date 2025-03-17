#include "../include/Map.h"

Map::Map() {}


void Map::debug_reprojection(std::vector<MapPoint *> local_map, std::vector<MapPoint *> out_map, KeyFrame *first_kf, int window) {
    std::vector<cv::KeyPoint> map_point_matched;
    for (MapPoint *mp : out_map) {
        int out = mp->reproject_map_point(first_kf, window, orb_descriptor_value);
        if (out == -1) continue;
        map_point_matched.push_back(first_kf->features[out].get_key_point());
    }
    std::cout << local_map.size() << " dimensiune initiala local map\n";
    std::cout << out_map.size() << " map points proiectate pe imagine\n";
    std::cout <<  map_point_matched.size() << " mapp points cu orb descriptors matched\n";
    cv::Mat img2, img3, img4;
    cv::drawKeypoints(first_kf->frame, first_kf->get_all_keypoints(), img2, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT); //albastru
    cv::drawKeypoints(img2, map_point_matched, img3, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT); //verde
    // cv::drawKeypoints(img3, map_point_matched, img4, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT); // rosu
    cv::imshow("Display window", img3);
    cv::waitKey(0);
}


std::vector<MapPoint *> Map::compute_map_points(KeyFrame *frame)
{
    std::vector<MapPoint *> current_points_found;
    for (int i = 0; i < frame->features.size(); i++)
    {
        cv::KeyPoint kp = frame->features[i].kp;
        float dd = frame->compute_depth_in_keypoint(kp);
        if (dd <= 0) continue;
        MapPoint *mp = new MapPoint(frame, i, dd);
        current_points_found.push_back(mp);
        frame->features[i].set_map_point(mp);
    }
    return current_points_found; 
}

Map::Map(KeyFrame *first_kf, Config cfg)
{
    this->orb_descriptor_value = orb_descriptor_value;
    std::vector<MapPoint *> kf_map_points = this->compute_map_points(first_kf);
    this->map_points.push_back(kf_map_points);
    this->graph.push_back(std::pair<KeyFrame *, std::unordered_map<KeyFrame *, int>>(first_kf, {}));
    debug_reprojection(kf_map_points, kf_map_points, first_kf, 15);
    std::cout << "SFARSIT INITIALIZARE\n";
}

std::vector<MapPoint *> Map::get_reprojected_map_points(KeyFrame *curr_frame, KeyFrame *reference_kf)
{
    if (this->map_points[reference_kf->idx].empty()) return {};
    std::vector<MapPoint *> out;
    for (MapPoint *mp : this->map_points[reference_kf->idx])
    {
        if (mp->map_point_belongs_to_keyframe(curr_frame)) out.push_back(mp);
    }
    return out;
}

KeyFrame *Map::get_reference_keyframe(KeyFrame *kf)
{
    KeyFrame *last_kf = this->graph.back().first;
    int last_idx = last_kf->idx;
    int max = -1;
    int reference_idx = last_idx;
    for (int i = 0; i < this->KEYFRAMES_WINDOW; i++)
    {
        int current_idx = last_idx - i;
        if (current_idx < 0)
            break;
        std::vector<MapPoint *> reprojected_map_points = get_reprojected_map_points(kf, this->graph[current_idx].first);
        if (reprojected_map_points.size() > max)
        {
            reference_idx = current_idx;
            max = reprojected_map_points.size();
        }
    }
    return this->graph[reference_idx].first;
}

std::vector<MapPoint *> Map::compute_local_map(KeyFrame *kf)
{
    KeyFrame *reference_kf = get_reference_keyframe(kf);
    std::unordered_map<KeyFrame *, int> reference_kf_neighbours = this->graph[reference_kf->idx].second;
    std::vector<MapPoint *> out = this->map_points[reference_kf->idx]; 
    for (std::pair<KeyFrame *, int> graph_edge : reference_kf_neighbours)
    {
        std::vector<MapPoint *> map_points_for_keyframe = this->map_points[graph_edge.first->idx]; 
        out.insert(out.end(), map_points_for_keyframe.begin(), map_points_for_keyframe.end());
    }
    return out;
}

// INCOMPLET, vor exista map point-uri duplicate
std::pair<std::vector<MapPoint *>, std::vector<cv::KeyPoint>> Map::track_local_map(KeyFrame *curr_kf, int window)
{
    std::vector<MapPoint *> local_map = this->compute_local_map(curr_kf);
    std::vector<MapPoint *> out_map;
    std::vector<cv::KeyPoint> kps;
    for (MapPoint *mp : local_map)
    {
        int idx = mp->reproject_map_point(curr_kf, window, this->orb_descriptor_value);
        if (idx == -1) continue;
        out_map.push_back(mp);
        kps.push_back(curr_kf->features[idx].get_key_point());
    }
    debug_reprojection(local_map, out_map, curr_kf, window);
    return std::pair<std::vector<MapPoint *>, std::vector<cv::KeyPoint>>(out_map, kps);
}