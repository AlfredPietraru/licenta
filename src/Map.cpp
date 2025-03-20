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

std::vector<MapPoint *> Map::compute_map_points(KeyFrame *frame)
{
    int null_values = 0;
    int negative_depth = 0;
    std::vector<MapPoint *> current_points_found;
    for (int i = 0; i < frame->features.size(); i++)
    {
        if (frame->features[i].get_map_point() != nullptr) {
            null_values++;
            continue;
        }
        float dd = frame->compute_depth_in_keypoint(frame->features[i].kp);
        if (dd <= 0) {
            negative_depth++;
            continue;  
        } 
        MapPoint *mp = new MapPoint(frame, i, dd);
        current_points_found.push_back(mp);
        frame->features[i].set_map_point(mp);
        frame->map_points.insert(mp);
    }
    std::cout << current_points_found.size() << " " << frame->features.size() << " " << null_values << " " << negative_depth << " debug compute map points\n"; 
    frame->nr_map_points = current_points_found.size();
    if (current_points_found.size() == 0) {
        std::cout << "CEVA NU E BINE NU S-AU CREAT PUNCTELE\n";
    }
    std::cout << current_points_found.size() << " puncte create\n";
    return current_points_found; 
}

void Map::add_new_keyframe(KeyFrame *new_kf) {
    this->compute_map_points(new_kf);
    std::unordered_map<KeyFrame*, int> edges_new_keyframe;
    if (this->keyframes.size() == 0) {
        this->keyframes.push_back(new_kf);
        this->graph.insert({new_kf, edges_new_keyframe});
        return;
    }

    int start_idx = (this->keyframes.size() > this->KEYFRAMES_WINDOW) ? this->keyframes.size() - this->KEYFRAMES_WINDOW : 0;
    std::cout << start_idx << " acesta a fost start_idx\n";
    for (int i = start_idx; i < this->keyframes.size(); i++) {
        KeyFrame *current_kf = this->keyframes[i];
        if (current_kf == nullptr) {
            std::cout << "A FOST NULL current_kf cand adaugam keyframe\n";
            continue;
        } 
        int common_values = this->matcher->get_number_common_mappoints_between_keyframes(new_kf, current_kf); 
        if (common_values < 15) {
            std::cout << "AU FOST GASITE MAI PUTIN DE 15 PUNCTE COMUNE INTRE FRAME-uri\n";
        }
        if (graph.find(current_kf) == graph.end()) {
            std::cout << " NU A FOST GASIT KEYFRAME-ul in graph\n";
            continue;
        }
        graph[current_kf].insert({new_kf, common_values});
        edges_new_keyframe.insert({current_kf, common_values});
    }   
    this->keyframes.push_back(new_kf);
    std::cout << this->keyframes.size() << " a facut adaugarea unui nou punct\n";
    this->graph.insert({new_kf, edges_new_keyframe});
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
    std::cout << this->keyframes.size() << " nr de keyframe-uri avute\n";
    for (int i = start_idx; i < this->keyframes.size(); i++)
    {
        std::cout << this->keyframes[i]->idx << " index curent analizat\n";
        std::vector<MapPoint *> reprojected_map_points = this->matcher->get_reprojected_map_points(kf, this->keyframes[i]);
        if (reprojected_map_points.size() == 0) {
            std::cout << "NU E BINE SA NU EXISTE NICIUN PUNCT NEPROIECTAT";
            continue;
        }
        if (reprojected_map_points.size() > max)
        {
            reference_idx = i;
            max = reprojected_map_points.size();
        }
    }
    return this->keyframes[reference_idx];
}

std::vector<MapPoint *> Map::compute_local_map(KeyFrame *current_frame)
{
    KeyFrame *reference_kf = get_reference_keyframe(current_frame);
    std::cout <<  reference_kf->idx << " compute local map reference_idx gasit\n";
    if (this->graph.find(reference_kf) == this->graph.end()) {
        std::cout << " CEVA NU E BINE NU GASESTE KEY FRAME IN COMPUTE LOCAL MAP\n";
    }
    std::vector<MapPoint *> out = reference_kf->return_map_points();
    std::cout <<  this->graph[reference_kf].size() << " nr de vecini asociati\n";
    for (std::pair<KeyFrame *, int> graph_edge : this->graph[reference_kf])
    {
        KeyFrame *curr_kf = graph_edge.first;
        if (curr_kf == nullptr)  {
            std::cout << " nu e bine ca e null\n";
            continue;
        }
        std::vector<MapPoint *> map_points_for_keyframe = curr_kf->return_map_points();  
        out.insert(out.end(), map_points_for_keyframe.begin(), map_points_for_keyframe.end());
    }
    return out;
}

// INCOMPLET, vor exista map point-uri duplicate
std::unordered_map<MapPoint *, Feature*> Map::track_local_map(KeyFrame *curr_kf, int window, KeyFrame *reference_kf)
{
    std::unordered_map<MapPoint *, Feature*> out;
    std::vector<MapPoint *> local_map = this->compute_local_map(curr_kf);
    for (MapPoint *mp : local_map)
    {
        int idx = mp->reproject_map_point(curr_kf, window, this->orb_descriptor_value);
        if (idx == -1) continue;
        out.insert(std::pair<MapPoint*, Feature*>(mp, &curr_kf->features[idx]));
    }
    // this->matcher->debug_reprojection(local_map, out, curr_kf, window, this->orb_descriptor_value);
    this->debug_map(reference_kf);
    return out;
}