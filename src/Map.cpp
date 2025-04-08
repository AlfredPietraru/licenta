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
        std::cout << it->first->current_idx << " " << it->second << " idx keyframe si numarul de conexiuni";
    }
}



void Map::add_first_keyframe(KeyFrame *new_kf) {
    this->compute_map_points(new_kf);
    this->keyframes.push_back(new_kf);
    this->graph.insert({new_kf, std::unordered_map<KeyFrame*, int>()});
}
 
void Map::add_new_keyframe(KeyFrame *new_kf) {
    this->compute_map_points(new_kf);
    std::unordered_map<KeyFrame*, int> edges_new_keyframe;
    
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
}

Map::Map(Orb_Matcher orb_matcher_cfg) {
    this->matcher = new OrbMatcher(orb_matcher_cfg);
}

KeyFrame *Map::get_reference_keyframe(KeyFrame *kf)
{
    int start_idx = (this->keyframes.size() > this->KEYFRAMES_WINDOW) ? this->keyframes.size() - this->KEYFRAMES_WINDOW : 0;
    int max = -1;
    int reference_idx = start_idx;
    for (int i = start_idx; i < this->keyframes.size(); i++)
    {
        int out = this->matcher->reproject_map_points(kf, this->keyframes[i]->map_points, 7);
        if (out == 0) continue;
        if (out > max)
        {
            reference_idx = i;
            max = out;
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
    return out;
}


void Map::compute_map_points(KeyFrame *kf)
{
    int map_points_associated = 0;
    int negative_depth = 0;
    int close_map_points = 0;
    int far_map_poins = 0;
    Eigen::Vector3d camera_center = kf->compute_camera_center();
    for (int i = 0; i < kf->features.size(); i++)
    {
        if (kf->features[i].get_map_point() != nullptr) {
            map_points_associated++;
            continue;
        }
        double depth = kf->compute_depth_in_keypoint(kf->features[i].kp);
        if (depth <= 1e-6) {
            negative_depth++;
            continue;  
        } 
        Eigen::Vector4d wcoord = kf->fromImageToWorld(i);
        MapPoint *mp = new MapPoint(kf, kf->features[i].kp, camera_center, wcoord,  kf->orb_descriptors.row(i), i);
        kf->features[i].set_map_point(mp);
        kf->map_points.insert(mp);
    }
    if (kf->map_points.size() == 0) {
        std::cout << "CEVA NU E BINE NU S-AU CREAT PUNCTELE\n";
    }
    std::cout << "\n";
    std::cout << kf->map_points.size() << " " << kf->features.size() << " " << map_points_associated << " " << negative_depth << " debug compute map points\n";  
}

// de adaugat reference keyframe
std::unordered_map<MapPoint *, Feature*> Map::track_local_map(KeyFrame *curr_kf)
{
    std::unordered_set<MapPoint*> local_map = this->compute_local_map(curr_kf);

    int window = curr_kf->current_idx > 2 ? 3 : 5;  
    return this->matcher->match_frame_map_points(curr_kf, local_map, window);
}

