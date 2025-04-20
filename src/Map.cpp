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

std::unordered_set<MapPoint*> Map::get_all_map_points() {
    std::unordered_set<MapPoint*> out;
    for (int i = 0; i < keyframes.size(); i++) {
        out.insert(keyframes[i]->map_points.begin(), keyframes[i]->map_points.end());
    }
    return out;
}


void Map::add_first_keyframe(KeyFrame *kf) {
    Eigen::Vector3d camera_center = kf->compute_camera_center_world();
    for (int i = 0; i < kf->features.size(); i++)
    {
        if (kf->features[i].get_map_point() != nullptr || kf->features[i].depth <= 1e-6) continue;
        Eigen::Vector4d wcoord = kf->fromImageToWorld(i);
        MapPoint *mp = new MapPoint(kf, kf->features[i].kpu, camera_center, wcoord, kf->features[i].descriptor);
        kf->features[i].set_map_point(mp, mp->orb_descriptor);
        kf->map_points.insert(mp);
        kf->mp_correlations.insert({mp, &kf->features[i]});
        mp->increase_how_many_times_seen();
        mp->increase_number_associations();
    }
    this->keyframes.push_back(kf);
    this->graph.insert({kf, std::unordered_map<KeyFrame*, int>()});
    this->local_map = kf->map_points;
    kf->compute_bow_representation();
}
 
void Map::add_new_keyframe(KeyFrame *new_kf) {
    std::unordered_map<KeyFrame*, int> edges_new_keyframe;
    
    int start_idx = (this->keyframes.size() > this->KEYFRAMES_WINDOW) ? this->keyframes.size() - this->KEYFRAMES_WINDOW : 0;
    for (int i = start_idx; i < this->keyframes.size(); i++) {
        KeyFrame *current_kf = this->keyframes[i];
        if (current_kf == nullptr) {
            std::cout << "A FOST NULL current_kf cand adaugam keyframe\n";
            continue;
        } 
        int common_values = get_number_common_mappoints_between_keyframes(new_kf, current_kf); 
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
    new_kf->compute_bow_representation();
    for (auto it = new_kf->mp_correlations.begin(); it != new_kf->mp_correlations.end(); it++) {
        MapPoint *mp = it->first;
        Feature *f = it->second;
        mp->add_observation_map_point(new_kf, f->descriptor, new_kf->compute_camera_center_world());
    }
    // de facut update la map points
}


int Map::get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2)
{
    int out = 0;
    for (auto it = kf1->map_points.begin(); it != kf1->map_points.end(); it++)
    {
        if (kf2->map_points.find(*it) != kf2->map_points.end()) out++;
    }
    return out;
}

Map::Map(Orb_Matcher orb_matcher_cfg) {
    this->matcher = new OrbMatcher(orb_matcher_cfg);
}


std::unordered_set<KeyFrame*> Map::get_local_keyframes(KeyFrame *kf) {
    if (this->graph.find(kf) == this->graph.end()) {
        std::cout << "NU EXISTA KEYFRAME-ul IN GRAPH\n";
        return {};
    }
    std::unordered_map<KeyFrame*, int> neighbours = this->graph[kf];
    std::unordered_set<KeyFrame*> out;
    for (auto it = neighbours.begin(); it != neighbours.end(); it++) {
        out.insert(it->first);
    }
    return out;
}

void Map::track_local_map(std::unordered_map<MapPoint *, Feature*> &matches, KeyFrame *curr_kf, KeyFrame *reference_kf)
{
    if (this->graph.find(reference_kf) == this->graph.end()) {
        std::cout << " \nCEVA NU E BINE NU GASESTE KEY FRAME IN COMPUTE LOCAL MAP\n";
        return;
    }
    int window = curr_kf->current_idx > 2 ? 3 : 5;  
    this->matcher->match_frame_map_points(matches, curr_kf, local_map, window);
}

