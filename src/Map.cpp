#include "../include/Map.h"
int ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

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
    for (long unsigned int i = 0; i < keyframes.size(); i++) {
        out.insert(keyframes[i]->map_points.begin(), keyframes[i]->map_points.end());
    }
    return out;
}


void Map::add_first_keyframe(KeyFrame *kf) {
    Eigen::Vector3d camera_center = kf->compute_camera_center_world();
    for (long unsigned int i = 0; i < kf->features.size(); i++)
    {
        if (kf->features[i].get_map_point() != nullptr || kf->features[i].depth <= 1e-6) continue;
        Eigen::Vector4d wcoord = kf->fromImageToWorld(i);
        MapPoint *mp = new MapPoint(kf, kf->features[i].kpu, camera_center, wcoord, kf->features[i].descriptor);
        add_map_point_to_keyframe(kf, &kf->features[i], mp);
    }

    this->keyframes.push_back(kf);
    this->graph.insert({kf, std::unordered_map<KeyFrame*, int>()});
    this->local_map = kf->map_points;
    kf->compute_bow_representation();
}

void Map::add_map_point_to_keyframe(KeyFrame *kf, Feature *f, MapPoint *mp) {
    if (f != &kf->features[f->idx]) {
        std::cout << "NU AM SETAT FEATURE-ul CORECT PENTRU ADAUGARE\n";
        return;
    }
    kf->remove_outlier_element(mp);
    MapPoint* old_mp = f->get_map_point();
    int current_hamming_distance = ComputeHammingDistance(mp->orb_descriptor, f->descriptor);
    if (old_mp == nullptr) {
        f->set_map_point(mp, current_hamming_distance);
        kf->mp_correlations.insert({mp, f});
        kf->map_points.insert(mp);
        mp->increase_number_associations();
        return;
    }
    if (current_hamming_distance >= f->curr_hamming_dist) return;
    f->set_map_point(mp, current_hamming_distance);
    kf->mp_correlations.erase(old_mp);
    kf->map_points.erase(old_mp);
    old_mp->decrease_number_associations();
    kf->mp_correlations.insert({mp, f});
    kf->map_points.insert(mp);
    mp->increase_number_associations();
}

void Map::add_keyframe_reference_to_map_point(MapPoint *mp, KeyFrame *kf) {
    if (kf->map_points.find(mp) ==  kf->map_points.end()) {
        std::cout << "PUNCTUL NICI MACAR NU SE REGASESTE IN KEYFRAME NU E BINE\n";
        return;
    }
    if (kf->mp_correlations.find(mp) == kf->mp_correlations.end()) {
        std::cout << "NU EXISTA PUNCTUL IN ASOCIERI\n";
    }
    mp->keyframes.insert(kf);
    mp->increase_number_associations();
    mp->compute_distinctive_descriptor(kf->mp_correlations[mp]->descriptor);
    mp->compute_view_direction(kf->compute_camera_center_world());
    std::vector<Eigen::Vector3d> centers;
    for (KeyFrame *kf : mp->keyframes) {
        centers.push_back(kf->compute_camera_center_world());
    }
    mp->compute_distance(centers);
}

void Map::remove_map_point_from_keyframe(KeyFrame *kf, MapPoint *mp) {
    if (kf->mp_correlations.find(mp) == kf->mp_correlations.end()) {
        std::cout << kf->current_idx << " " << kf->mp_correlations.size() << " " << kf->map_points.size() << "\n";
        std::cout << "NU A EXISTAT PUNCTUL IN CORELATII\n";
            return;
        }
        if (kf->map_points.find(mp) == kf->map_points.end()) {
            std::cout << kf->current_idx << " " << kf->mp_correlations.size() << " " << kf->map_points.size() << "\n";
            std::cout << "NU A EXISTAT PUNCTUL IN MAP POINTS, NU SUNT SINCRONIZATE\n";
            return;
        }
    
        Feature *f = kf->mp_correlations[mp];
        // mai trebuie de lucrat aici
        f->unmatch_map_point();
        kf->mp_correlations.erase(mp);
        kf->map_points.erase(mp);
        mp->decrease_number_associations();    
}
 
void Map::add_new_keyframe(KeyFrame *new_kf) {
    std::unordered_map<KeyFrame*, int> edges_new_keyframe;
    
    int start_idx = ((int)this->keyframes.size() > this->KEYFRAMES_WINDOW) ? this->keyframes.size() - this->KEYFRAMES_WINDOW : 0;
    for (long unsigned int i = start_idx; i < this->keyframes.size(); i++) {
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
        Map::add_keyframe_reference_to_map_point(it->first, new_kf);
    }
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

void Map::track_local_map(std::unordered_map<MapPoint *, Feature*> &matches, KeyFrame *kf, KeyFrame *reference_kf)
{
    if (this->graph.find(reference_kf) == this->graph.end()) {
        std::cout << " \nCEVA NU E BINE NU GASESTE KEY FRAME IN COMPUTE LOCAL MAP\n";
        return;
    }

    int window = kf->current_idx > 2 ? 3 : 5;  
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center_world();
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector3d point_camera_coordinates;
    
    for (MapPoint *mp : reference_kf->map_points) {
        if (kf->check_map_point_outlier(mp)) continue;
        point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < kf->minX || point_camera_coordinates(0) > kf->maxX - 1) continue;
        if (point_camera_coordinates(1) < kf->minY || point_camera_coordinates(1) > kf->maxY - 1) continue;
        if (point_camera_coordinates(2) < 1e-6) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        double distance = camera_to_map_view_ray.norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;
        mp->increase_how_many_times_seen(); 

        float radius = dot_product >= 0.998 ? 2.5 : 4;
        radius *= window;
        int predicted_scale = mp->predict_image_scale(distance);
        int scale_of_search = radius * pow(1.2, predicted_scale);        
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);

        int lowest_idx = -1;
        int lowest_dist = 256;
        int lowest_level = -1;

        int second_lowest_dist = 256;
        int second_lowest_level = -1;
        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, predicted_scale - 1, predicted_scale + 1);
        if (kps_idx.size() == 0) continue;
        for (int idx : kps_idx)
        {
            if (kf->features[idx].stereo_depth > 1e-6) {
                double fake_rgbd = u - kf->K(0, 0) * 0.08 / point_camera_coordinates(2);
                float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
                if (er > scale_of_search) continue;
            }

            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            
            if (cur_hamm_dist < lowest_dist)
            {
                second_lowest_dist = lowest_dist;
                second_lowest_level = lowest_level;
                lowest_idx = idx;
                lowest_dist = cur_hamm_dist;
                lowest_level = kf->features[idx].get_undistorted_keypoint().octave;
            }
            else if (cur_hamm_dist < second_lowest_dist)
            {
                second_lowest_dist = cur_hamm_dist;
                second_lowest_level = kf->features[idx].get_undistorted_keypoint().octave;
            }
        }
        if (lowest_dist > 50)  continue;
        if(lowest_level == second_lowest_level && lowest_dist > 0.8 * second_lowest_dist) continue;
        matches.insert({mp, &kf->features[lowest_idx]});
    }
}

