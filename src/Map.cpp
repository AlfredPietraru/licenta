#include "../include/Map.h"

Map::Map() {}

void Map::debug_map(KeyFrame *kf) {
    if (this->graph.find(kf) == this->graph.end()) {
        std::cout << " NU A FOST GASIT KEYFRAME-ul\n";
        return;
    }
    std::unordered_map<KeyFrame*, int> edges = this->graph[kf];
    if (edges.size() == 0) {
        std::cout << "DIMENSIUNE EDGES ESTE 0, NOD IZOLAT NU E BINE\n\n\n";
        return;
    }
    for (auto it = edges.begin(); it != edges.end(); it++) {
        std::cout << it->first->idx << " " << it->second << " idx keyframe si numarul de conexiuni";
    }
}

void Map::add_new_keyframe(KeyFrame *new_kf) {
    new_kf->compute_map_points();
    std::unordered_map<KeyFrame*, int> edges_new_keyframe;
    if (this->keyframes.size() == 0) {
        this->keyframes.push_back(new_kf);
        this->graph.insert({new_kf, edges_new_keyframe});
        local_map = this->compute_local_map(new_kf);
        return;
    }
    
    int start_idx = (this->keyframes.size() > this->KEYFRAMES_WINDOW) ? this->keyframes.size() - this->KEYFRAMES_WINDOW : 0;
    for (int i = start_idx; i < this->keyframes.size(); i++) {
        KeyFrame *current_kf = this->keyframes[i];
        if (current_kf == nullptr) {
            std::cout << "A FOST NULL current_kf cand adaugam keyframe\n";
            continue;
        } 
        int common_values = this->matcher->get_number_common_mappoints_between_keyframes(new_kf, current_kf); 
        if (common_values < 15) {
            std::cout << "AU FOST GASITE MAI PUTIN DE 15 PUNCTE COMUNE INTRE FRAME-uri\n";
            continue;
        }
        if (graph.find(current_kf) == graph.end()) {
            std::cout << " NU A FOST GASIT KEYFRAME-ul in graph\n";
            continue;
        }
        graph[current_kf].insert({new_kf, common_values});
        edges_new_keyframe.insert({current_kf, common_values});
    }   
    this->keyframes.push_back(new_kf);
    this->graph.insert({new_kf, edges_new_keyframe});
    local_map = this->compute_local_map(new_kf);
}

Map::Map(OrbMatcher *matcher, KeyFrame *first_kf, Config cfg)
{
    this->matcher = matcher;
    this->orb_descriptor_value = cfg.orb_descriptor_value;
    this->add_new_keyframe(first_kf);
    std::cout << "SFARSIT INITIALIZARE\n";
}

KeyFrame *Map::get_reference_keyframe(KeyFrame *kf)
{
    int start_idx = (this->keyframes.size() > this->KEYFRAMES_WINDOW) ? this->keyframes.size() - this->KEYFRAMES_WINDOW : 0;
    int max = -1;
    int reference_idx = start_idx;
    for (int i = start_idx; i < this->keyframes.size(); i++)
    {
        std::unordered_map<MapPoint*, Feature*> reprojected_map_points = this->matcher->match_frame_map_points(kf, this->keyframes[i]->map_points);
        if (reprojected_map_points.size() == 0) continue;
        if (reprojected_map_points.size() > max)
        {
            reference_idx = i;
            max = reprojected_map_points.size();
        }
    }
    return this->keyframes[reference_idx];
}

std::unordered_set<MapPoint *> Map::compute_local_map(KeyFrame *current_frame)
{
    KeyFrame *reference_kf = get_reference_keyframe(current_frame);
    if (this->graph.find(reference_kf) == this->graph.end()) {
        std::cout << " CEVA NU E BINE NU GASESTE KEY FRAME IN COMPUTE LOCAL MAP\n";
    }
    std::unordered_set<MapPoint *> out = reference_kf->map_points;
    for (std::pair<KeyFrame *, int> graph_edge : this->graph[reference_kf])
    {
        KeyFrame *curr_kf = graph_edge.first;
        if (curr_kf == nullptr)  {
            std::cout << " nu e bine ca e null\n";
            continue;
        }
        std::unordered_set<MapPoint *> map_points_for_keyframe = curr_kf->map_points;  
        for (auto it = map_points_for_keyframe.begin(); it != map_points_for_keyframe.end(); it++) {
            out.insert(*it);
        }
    }
    std::cout << reference_kf->map_points.size() << " "  << out.size() << " dimensiune map points\n";
    return out;
}

void Map::track_local_map(KeyFrame *curr_kf, std::unordered_map<MapPoint *, Feature*>& matches,  int window)
{
    // return this->matcher->match_frame_map_points(curr_kf, local_map);
    curr_kf->currently_matched_points = matches.size();
    std::cout << curr_kf->currently_matched_points << " " << curr_kf->maximum_possible_map_points << " de testat cum evolueaza track local map\n";
    std::unordered_map<MapPoint *, Feature*> out;
    for (MapPoint *mp : local_map) {
        Eigen::Vector3d point_camera_coordinates =  curr_kf->fromWorldToImage(mp->wcoord);
        for (int i = 0; i < 3; i++) {
            if (point_camera_coordinates(i) < 0)  continue;
        }
        if (point_camera_coordinates(0) > curr_kf->depth_matrix.cols - 1) continue;
        if (point_camera_coordinates(1) > curr_kf->depth_matrix.rows - 1) continue;
        if (mp->dmax < point_camera_coordinates(2) || mp->dmin > point_camera_coordinates(2)) continue;
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        int min_hamm_dist = 10000;
        int cur_hamm_dist;
        int out_idx = -1;
        std::vector<int> kps_idx = curr_kf->get_vector_keypoints_after_reprojection(u, v, window);
        if (kps_idx.size() == 0) continue;
        for (int idx : kps_idx) {
            cur_hamm_dist = this->matcher->ComputeHammingDistance(mp->orb_descriptor, curr_kf->features[idx].descriptor);
            if (cur_hamm_dist < min_hamm_dist) {
                out_idx = idx;
                min_hamm_dist = cur_hamm_dist;
            }
        }
        if (min_hamm_dist > orb_descriptor_value || out_idx == -1) continue;
        // map point is valid
        if (matches.find(mp) == matches.end()) {
            matches.insert({mp, &curr_kf->features[out_idx]});
            curr_kf->currently_matched_points++;
        } else {
            
        }
        if (curr_kf->currently_matched_points == curr_kf->maximum_possible_map_points) break;
    }
    std::cout << curr_kf->currently_matched_points << " " << curr_kf->maximum_possible_map_points << " rezultat dupa track local map\n";
    // this->matcher->debug_reprojection(local_map, matches, curr_kf, window, this->orb_descriptor_value);
}