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

Map::Map(Orb_Matcher orb_matcher_cfg) {
    this->matcher = new OrbMatcher(orb_matcher_cfg);
}

Map::Map(OrbMatcher *matcher, KeyFrame *first_kf, Config cfg)
{
    this->matcher = matcher;
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
    return out;
}

std::unordered_map<MapPoint *, Feature*> Map::track_local_map(KeyFrame *curr_kf, std::unordered_map<MapPoint *, Feature*>& matches)
{
    std::unordered_set<MapPoint *> new_local_map;
    for (auto it = local_map.begin(); it != local_map.end(); it++) {
        if (matches.find(*it) == matches.end()) {
            new_local_map.insert(*it);
        }
    }
    return this->matcher->match_frame_map_points(curr_kf, new_local_map);
    // curr_kf->currently_matched_points = out.size();
    
    // this->matcher->debug_reprojection(local_map, matches, curr_kf, window, this->orb_descriptor_value);
}