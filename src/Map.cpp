#include "../include/Map.h"
int ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist = 0;

    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        unsigned int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

std::unordered_set<MapPoint *> Map::get_all_map_points()
{
    std::unordered_set<MapPoint *> out;
    for (long unsigned int i = 0; i < keyframes.size(); i++)
    {
        out.insert(keyframes[i]->map_points.begin(), keyframes[i]->map_points.end());
    }
    return out;
}

std::unordered_map<KeyFrame *, int> Map::get_keyframes_connected(KeyFrame *new_kf, int limit)
{
    std::unordered_map<KeyFrame *, int> edges_new_keyframe;
    for (MapPoint *mp : new_kf->map_points)
    {
        for (KeyFrame *kf : mp->keyframes)
        {
            if (edges_new_keyframe.find(kf) != edges_new_keyframe.end())
            {
                edges_new_keyframe[kf] += 1;
            }
            else
            {
                edges_new_keyframe[kf] = 1;
            }
        }
    }
    std::vector<KeyFrame *> to_remove;
    for (auto it = edges_new_keyframe.begin(); it != edges_new_keyframe.end(); it++)
    {
        if (it->second < limit)
            to_remove.push_back(it->first);
    }
    for (KeyFrame *kf : to_remove)
    {
        edges_new_keyframe.erase(kf);
    }
    return edges_new_keyframe;
}

std::vector<MapPoint*> Map::create_map_points_from_features(KeyFrame *kf)
{
    std::vector<MapPoint*> out;
    for (long unsigned int i = 0; i < kf->features.size(); i++)
    {
        if (kf->features[i].get_map_point() != nullptr)
            continue;
        double depth = kf->features[i].depth;
        if (depth <= 1e-6 || depth > 3.2)
            continue;
        Eigen::Vector4d wcoord = kf->fromImageToWorld(i);
        MapPoint *mp = new MapPoint(kf, i, kf->features[i].kpu, kf->camera_center_world, wcoord, kf->features[i].descriptor);
        out.push_back(mp);
        mp->increase_how_many_times_seen();
        bool was_addition_succesfull = add_map_point_to_keyframe(kf, &kf->features[i], mp);
        if (!was_addition_succesfull)
        {
            std::cout << "NU S-A PUTUT SA ADAUGE MAP POINT-ul\n";
        }
    }
    return out;
}

std::vector<MapPoint*> Map::add_new_keyframe(KeyFrame *new_kf)
{
    new_kf->reference_idx += 1;
    std::cout << new_kf->reference_idx << " acesta este indexul map point-ului\n";
    this->keyframes.push_back(new_kf);
    this->graph[new_kf] = std::unordered_map<KeyFrame *, int>();
    new_kf->compute_bow_representation();
    if (this->keyframes.size() == 1)
    {
        (void)this->create_map_points_from_features(new_kf);
        return std::vector<MapPoint*>();
    }
    
    for (auto it = new_kf->mp_correlations.begin(); it != new_kf->mp_correlations.end(); it++)
    {
        bool was_addition_succesfull = Map::add_keyframe_reference_to_map_point(it->first, new_kf->mp_correlations[it->first], new_kf);
        if (!was_addition_succesfull)
        {
            std::cout << "NU S-A PUTUT ADAUGA NOUL MAP POINT IN KEYFRAME IN MAP INAINTE\n";
        }
    }
    std::vector<MapPoint*> points_added = this->create_map_points_from_features(new_kf);
    std::unordered_map<KeyFrame *, int> edges_new_keyframe = this->get_keyframes_connected(new_kf, 15);
    for (auto it = edges_new_keyframe.begin(); it != edges_new_keyframe.end(); it++)
    {
        this->graph[it->first][new_kf] = it->second;
        this->graph[new_kf][it->first] = it->second;
    }
    return points_added;
}

bool Map::add_map_point_to_keyframe(KeyFrame *kf, Feature *f, MapPoint *mp)
{
    if (kf == nullptr || mp == nullptr || f == nullptr)
    {
        std::cout << "NU S-A PUTUT REALIZA OPERATIA CEVA ERA NULL\n";
        return false;
    }

    bool in_map_points = kf->map_points.find(mp) != kf->map_points.end();
    bool in_correlations = kf->mp_correlations.find(mp) != kf->mp_correlations.end();
    bool in_features = in_correlations ? (kf->mp_correlations[mp]->get_map_point() == mp) : in_correlations;
    if (in_map_points && in_correlations && in_features)
        return false;
    if (in_map_points != in_correlations || in_correlations != in_features)
    {
        std::cout << kf->map_points.size() << " " << kf->mp_correlations.size() << "\n";
        std::cout << in_map_points << " " << in_correlations << " " << in_features << "\n";
        std::cout << "NU S-A PUTUT ADAUGA MAP POINT IN KEYFRAME, KEYFRAME NESINCRONIZAT\n";
        return false;
    }

    kf->remove_outlier_element(mp);
    MapPoint *old_mp = f->get_map_point();
    int current_hamming_distance = ComputeHammingDistance(mp->orb_descriptor, f->descriptor);
    if (old_mp == nullptr)
    {
        mp->increase_number_associations(1);
        f->set_map_point(mp, current_hamming_distance);
        kf->mp_correlations.insert({mp, f});
        kf->map_points.insert(mp);
        return true;
    }
    if (current_hamming_distance >= f->curr_hamming_dist)
        return false;

    bool deletion_result = remove_map_point_from_keyframe(kf, old_mp);
    if (!deletion_result)
    {
        std::cout << "NU S-A PUTUT CAND FACEA ADAUGAREA ALTUI ELEMENT NU L-A PUTUT STERGE PE CEL VECHI\n";
        return false;
    }

    mp->increase_number_associations(1);
    f->set_map_point(mp, current_hamming_distance);
    kf->mp_correlations.insert({mp, f});
    kf->map_points.insert(mp);
    return true;
}

bool Map::remove_map_point_from_keyframe(KeyFrame *kf, MapPoint *mp)
{
    if (mp == nullptr || kf == nullptr)
    {
        std::cout << "NU S-A PUTUT REALIZA STERGEREA IN REMOVE MAP FROM KEYFRAME UNUL DINTRE ELEMENTE E NULL\n";
        return false;
    }

    bool not_in_map_points = kf->map_points.find(mp) == kf->map_points.end();
    bool not_in_correlations = kf->mp_correlations.find(mp) == kf->mp_correlations.end();
    bool not_in_features = not_in_correlations ? not_in_correlations : kf->mp_correlations[mp]->get_map_point() != mp;
    // bool already_found_in_keyframe = mp->find_keyframe(kf);

    if (not_in_map_points && not_in_correlations && not_in_features)
        return true;
    if (not_in_correlations != not_in_features || not_in_features != not_in_map_points)
    {
        std::cout << "KEYFRAME UL NU ESTE SINCRONIZAT LA STERGERE\n";
        return false;
    }

    mp->decrease_number_associations(1);
    Feature *f = kf->mp_correlations[mp];
    f->unmatch_map_point();
    kf->mp_correlations.erase(mp);
    kf->map_points.erase(mp);
    return true;
}

bool Map::add_keyframe_reference_to_map_point(MapPoint *mp, Feature *f, KeyFrame *kf)
{
    if (mp == nullptr || kf == nullptr)
    {
        std::cout << "NU S-A PUTUT REALIZA OPERATIA IN ADD KEYFRAMAE REFERENCE UNUL DINTRE ELEMENTE E NULL\n";
        return false;
    }

    bool not_in_map_points = kf->map_points.find(mp) == kf->map_points.end();
    bool not_in_correlations = kf->mp_correlations.find(mp) == kf->mp_correlations.end();
    bool not_in_features = not_in_correlations ? not_in_correlations : kf->mp_correlations[mp]->get_map_point() != mp;
    bool already_found_in_keyframe = mp->find_keyframe(kf);

    if (not_in_map_points && not_in_correlations && not_in_features)
    {
        std::cout << "NU S-A PUTUT REALIZA ADAUGAREA KEYFRAME LA REFERINTA MAP POINT NU EXISTA PUNCTUL IN KEYFRAME\n";
        return false;
    }
    if (not_in_map_points != not_in_correlations || not_in_features != not_in_correlations)
    {
        std::cout << "NU S-A PUTUT REALIZA ADAUGAREA KEYFRAME LA REFERINTA MISMATCH IN KEYFRAME\n";
        return false;
    }
    if (already_found_in_keyframe)
    {
        std::cout << "ESTE DEJA ADAUGAT CA REFERINTA\n";
        return false;
    }

    mp->add_observation(kf, f->idx, kf->camera_center_world, kf->mp_correlations[mp]->descriptor);
    return true;
}

bool Map::remove_keyframe_reference_from_map_point(MapPoint *mp, KeyFrame *kf)
{
    if (mp == nullptr || kf == nullptr)
    {
        std::cout << "NU S-A PUTUT REALIZA OPERATIA DE REMOVE KEYFRAME REFERENCE UNUL DINTRE ELEMENTE E NULL\n";
        return false;
    }
    if (!mp->find_keyframe(kf))
    {
        std::cout << "NU S-A PUTUT REALIZA OPERATIA DE REMOVE MAP POINT-ul NU EXISTA CA REFERINTA IN KEYFRAME\n";
        return false;
    }
    mp->remove_observation(kf);
    return true;
}

std::unordered_set<KeyFrame *> Map::get_local_keyframes(KeyFrame *kf)
{
    if (this->graph.find(kf) == this->graph.end())
    {
        std::cout << "NU EXISTA KEYFRAME-ul IN GRAPH\n";
        return {};
    }
    std::unordered_map<KeyFrame *, int> neighbours = this->graph[kf];
    std::unordered_set<KeyFrame *> out;
    for (auto it = neighbours.begin(); it != neighbours.end(); it++)
    {
        out.insert(it->first);
    }
    return out;
}

std::unordered_set<KeyFrame *> Map::get_till_second_degree_keyframes(KeyFrame *kf)
{
    std::unordered_set<KeyFrame *> first_degree_key_frames = this->get_local_keyframes(kf);
    std::unordered_set<KeyFrame *> second_degree_key_frames;

    second_degree_key_frames.insert(first_degree_key_frames.begin(), first_degree_key_frames.end());
    for (KeyFrame *kf : first_degree_key_frames)
    {
        std::unordered_set<KeyFrame *> current_kf_neighbours = this->get_local_keyframes(kf);
        second_degree_key_frames.insert(current_kf_neighbours.begin(), current_kf_neighbours.end());
    }
    if (second_degree_key_frames.find(kf) != second_degree_key_frames.end())
    {
        second_degree_key_frames.erase(kf);
    }
    return second_degree_key_frames;
}

bool Map::update_graph_connections(KeyFrame *kf1, KeyFrame *kf2)
{
    if (this->graph.find(kf1) == this->graph.end() || this->graph.find(kf2) == this->graph.end())
    {
        std::cout << "NU S-AU GASIT NODURILE IN GRAPH\n";
        return false;
    }
    int common_values = 0;
    for (MapPoint *mp : kf1->map_points)
    {
        if (kf2->map_points.find(mp) != kf2->map_points.end())
            common_values++;
    }
    if (common_values < 15)
    {
        if (this->graph[kf1].find(kf2) != this->graph[kf1].end())
        {
            this->graph[kf1].erase(kf2);
        }
        if (this->graph[kf2].find(kf1) != this->graph[kf2].end())
        {
            this->graph[kf2].erase(kf1);
        }
        return true;
    }
    this->graph[kf1][kf2] = common_values;
    this->graph[kf2][kf1] = common_values;
    return true;
}

void Map::track_local_map(KeyFrame *kf, KeyFrame *ref, std::unordered_set<KeyFrame *> &keyframes_already_found)
{
    if (ref == nullptr)
    {
        std::cout << "ESTE REFERENCE FRAME GOL IN TRACK LOCAL MAP\n";
        return;
    }
    if (this->graph.find(ref) == this->graph.end())
    {
        std::cout << " \nCEVA NU E BINE NU GASESTE KEY FRAME IN COMPUTE LOCAL MAP\n";
        return;
    }

    std::unordered_set<KeyFrame *> second_degree_key_frames = this->get_till_second_degree_keyframes(ref);
    second_degree_key_frames.insert(keyframes_already_found.begin(), keyframes_already_found.end());
    second_degree_key_frames.insert(ref);

    int window = kf->current_idx > 2 ? 3 : 5;
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector4d point_camera_coordinates;
    double u, v, d;
    for (KeyFrame *local_map_kf : second_degree_key_frames)
    {
        for (MapPoint *mp : local_map_kf->map_points)
        {
            if (kf->check_map_point_outlier(mp))
                continue;
            point_camera_coordinates = kf->mat_camera_world * mp->wcoord;
            d = point_camera_coordinates(2);
            if (d <= 1e-6)
                continue;
            u = kf->K(0, 0) * point_camera_coordinates(0) / d + kf->K(0, 2);
            v = kf->K(1, 1) * point_camera_coordinates(1) / d + kf->K(1, 2);
            if (u < kf->minX || u > kf->maxX - 1)
                continue;
            if (v < kf->minY || v > kf->maxY - 1)
                continue;

            camera_to_map_view_ray = (mp->wcoord_3d - kf->camera_center_world);
            double distance = camera_to_map_view_ray.norm();
            if (distance < mp->dmin || distance > mp->dmax)
                continue;

            camera_to_map_view_ray.normalize();
            double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
            if (dot_product < 0.5)
                continue;
            mp->increase_how_many_times_seen();

            float radius = dot_product >= 0.998 ? 2.5 : 4;
            radius *= window;
            int predicted_scale = mp->predict_image_scale(distance);
            int scale_of_search = radius * kf->POW_OCTAVE[predicted_scale];

            int lowest_idx = -1;
            int lowest_dist = 256;
            int lowest_level = -1;

            int second_lowest_dist = 256;
            int second_lowest_level = -1;
            std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, predicted_scale - 1, predicted_scale + 1);
            if (kps_idx.size() == 0)
                continue;
            for (int idx : kps_idx)
            {
                if (kf->features[idx].stereo_depth >= 1e-6)
                {
                    double fake_rgbd = u - kf->K(0, 0) * 0.08 / d;
                    float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
                    if (er > scale_of_search)
                        continue;
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
            if (lowest_dist > 50)
                continue;
            if (lowest_level == second_lowest_level && lowest_dist > 0.8 * second_lowest_dist)
                continue;
            Map::add_map_point_to_keyframe(kf, &kf->features[lowest_idx], mp);
        }
    }
}

bool Map::replace_map_point(MapPoint *old_mp, MapPoint *new_mp)
{
    if (old_mp == new_mp) return true;
    std::vector<KeyFrame*> copy_keyframes(old_mp->keyframes.begin(), old_mp->keyframes.end());
    for (KeyFrame *kf : copy_keyframes) {
        Feature *old_map_point_feature = kf->mp_correlations[old_mp];
        Map::remove_map_point_from_keyframe(kf, old_mp);
        Map::remove_keyframe_reference_from_map_point(old_mp, kf);
        if (!kf->check_map_point_in_keyframe(new_mp)) {
            Map::add_map_point_to_keyframe(kf, old_map_point_feature, new_mp);
            Map::remove_keyframe_reference_from_map_point(new_mp, kf);
        }
    }
    return true;
}

bool Map::new_map_point_better(Feature *f, MapPoint *new_map_point) {
    MapPoint *old_mp = f->get_map_point();
    if (old_mp == nullptr) return true;
    int dist = ComputeHammingDistance(f->descriptor, new_map_point->orb_descriptor);
    return dist < f->curr_hamming_dist;
}
