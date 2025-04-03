#include "../include/OrbMatcher.h"

int inline ::OrbMatcher::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2)
{
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++)
    {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i);
        distance += __builtin_popcount(v);
    }
    return distance;
}

void sort_values(std::vector<std::pair<MapPoint *, Feature *>> &map_points)
{
    for (int i = 0; i < map_points.size() - 1; i++)
    {
        for (int j = i + 1; j < map_points.size(); j++)
        {
            if (map_points[i].second->depth > map_points[j].second->depth)
            {
                std::pair<MapPoint *, Feature *> current = map_points[i];
                map_points[i] = map_points[j];
                map_points[j] = current;
            }
        }
    }
}

std::unordered_map<MapPoint *, Feature *> OrbMatcher::checkOrientation(std::unordered_map<MapPoint *, Feature *> &correlation_current_frame,
                                                                       std::unordered_map<MapPoint *, Feature *> &correlation_prev_frame)
{
    std::vector<std::vector<MapPoint *>> histogram(30, std::vector<MapPoint *>());
    const float factor = 1.0f / 30;
    for (auto it = correlation_current_frame.begin(); it != correlation_current_frame.end(); it++)
    {
        MapPoint *mp = it->first;
        if (correlation_prev_frame.find(mp) == correlation_prev_frame.end())
        {
            correlation_current_frame.erase(mp);
            continue;
        }
        cv::KeyPoint current_kp = it->second->get_key_point();
        cv::KeyPoint prev_kp = correlation_prev_frame[mp]->get_key_point();
        float rot = prev_kp.angle - current_kp.angle;
        if (rot < 0.0)
            rot += 360.0f;
        int bin = round(rot * factor);
        if (bin == 30)
            bin = 0;
        if (bin < 0 || bin > 30)
        {
            std::cout << "CEVA NU E BINE CHECK ORIENTATION\n";
            exit(1);
        }
        histogram[bin].push_back(it->first);
    }
    int maxval1 = 0, maxval2 = 0, maxval3 = 0;
    int maxidx1 = -1, maxidx2 = -1, maxidx3 = -1;
    for (int i = 0; i < histogram.size(); i++)
    {
        if (histogram[i].size() > maxval1)
        {
            maxidx3 = maxidx2;
            maxidx2 = maxidx1;
            maxidx1 = i;
            maxval3 = maxval2;
            maxval2 = maxval1;
            maxval1 = histogram[i].size();
            continue;
        }
        else if (histogram[i].size() > maxval2)
        {
            maxidx3 = maxidx2;
            maxidx2 = i;
            maxval3 = maxval2;
            maxval2 = histogram[i].size();
            continue;
        }
        else if (histogram[i].size() > maxval3)
        {
            maxidx3 = i;
            maxval3 = histogram[i].size();
        }
    }
    for (int i = 0; i < histogram.size(); i++)
    {
        std::cout << histogram[i].size() << " ";
    }
    std::cout << "\n";
    std::cout << "indexi gasiti " << maxidx1 << " " << maxidx2 << " " << maxidx3 << "\n";
    std::cout << "valori maxime gasite " << histogram[maxidx1].size() << " " << histogram[maxidx2].size() << " " << histogram[maxidx3].size() << "\n";
    for (int i = 0; i < histogram.size(); i++)
    {
        if (i == maxidx1 || i == maxidx2 || i == maxidx3)
            continue;
        for (int j = 0; j < histogram[i].size(); j++)
        {
            correlation_current_frame.erase(histogram[i][j]);
        }
    }
    return correlation_current_frame;
}

std::unordered_map<MapPoint *, Feature *> OrbMatcher::match_frame_map_points(KeyFrame *kf, KeyFrame *prev_kf)
{
    std::unordered_map<MapPoint *, Feature *> out;
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    double window_size = 2.5;
    Eigen::Vector3d camera_to_map_view_ray;
    std::unordered_map<MapPoint *, Feature *> prev_frame_correlations = prev_kf->return_map_points_keypoint_correlation();
    for (auto it = prev_frame_correlations.begin(); it != prev_frame_correlations.end(); it++)
    {
        MapPoint *mp = it->first;
        if (mp == nullptr || mp->is_outlier) continue;
        Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        for (int i = 0; i < 3; i++)
        {
            if (point_camera_coordinates(i) < 0)
                continue;
        }
        if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1)
            continue;
        if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1)
            continue;

        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;

        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);

        int lowest_dist = 256;
        int lowest_idx = -1;
        int lowest_level = -1;
        int second_lowest_dist = 256;
        int second_lowest_level = -1;
        int second_lowest_idx = -1;

        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, this->window);
        if (kps_idx.size() == 0)
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, 2 * this->window);

        for (int idx : kps_idx)
        {
            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            if (kf->features[idx].get_map_point() != nullptr)
                continue; // verifica sa fie asociat unui singur map point
            if (kf->features[idx].stereo_depth > 0)
            {
                float er = fabs(point_camera_coordinates(2) - kf->features[idx].stereo_depth);
                if (er > window_size * mp->predicted_scale)
                    continue;
            }
            if (cur_hamm_dist < lowest_dist)
            {
                second_lowest_dist = lowest_dist;
                second_lowest_idx = lowest_idx;
                second_lowest_level = lowest_level;
                lowest_idx = idx;
                lowest_dist = cur_hamm_dist;
                lowest_level = kf->features[idx].kp.octave;
            }
            else if (cur_hamm_dist < second_lowest_dist)
            {
                second_lowest_dist = cur_hamm_dist;
                second_lowest_idx = idx;
                second_lowest_level = kf->features[idx].kp.octave;
            }
        }

        if (lowest_dist > this->orb_descriptor_value)
            continue;
        if (lowest_level == second_lowest_level && lowest_dist > this->ration_first_second_match * second_lowest_dist)
            continue;
        if (mp->is_safe_to_use)
            out.insert({mp, &kf->features[lowest_idx]});
    }
    std::cout << out.size() << " atatea map points calculate initial inainte de verificare orientarii\n";
    this->checkOrientation(out, prev_frame_correlations);
    return out;
}

std::unordered_map<MapPoint *, Feature *> OrbMatcher::match_frame_map_points(KeyFrame *kf, std::unordered_set<MapPoint *> map_points)
{
    std::unordered_map<MapPoint *, Feature *> out;
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    double window_size = 2.5;
    Eigen::Vector3d camera_to_map_view_ray;
    for (auto it = map_points.begin(); it != map_points.end(); it++)
    {
        MapPoint *mp = *it;
        Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        for (int i = 0; i < 3; i++)
        {
            if (point_camera_coordinates(i) <= 0)
                continue;
        }
        if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1)
            continue;
        if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1)
            continue;

        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5)
            continue;

        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);

        int lowest_dist = 256;
        int lowest_idx = -1;
        int lowest_level = -1;
        int second_lowest_dist = 256;
        int second_lowest_level = -1;
        int second_lowest_idx = -1;

        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, this->window);
        if (kps_idx.size() == 0)
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, 2 * this->window);

        for (int idx : kps_idx)
        {
            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            if (kf->features[idx].get_map_point() != nullptr)
                continue; // verifica sa fie asociat unui singur map point
            if (kf->features[idx].stereo_depth > 0)
            {
                float er = fabs(point_camera_coordinates(2) - kf->features[idx].stereo_depth);
                if (er > window_size * mp->predicted_scale)
                    continue;
            }
            if (cur_hamm_dist < lowest_dist)
            {
                second_lowest_dist = lowest_dist;
                second_lowest_idx = lowest_idx;
                second_lowest_level = lowest_level;
                lowest_idx = idx;
                lowest_dist = cur_hamm_dist;
                lowest_level = kf->features[idx].kp.octave;
            }
            else if (cur_hamm_dist < second_lowest_dist)
            {
                second_lowest_dist = cur_hamm_dist;
                second_lowest_idx = idx;
                second_lowest_level = kf->features[idx].kp.octave;
            }
        }

        if (lowest_dist > this->orb_descriptor_value)
            continue;
        if (lowest_level == second_lowest_level && lowest_dist > this->ration_first_second_match * second_lowest_dist)
            continue;
        if (mp->is_safe_to_use)
            out.insert({mp, &kf->features[lowest_idx]});
    }
    // de vazut cu orientarea aici
    return out;
}

int OrbMatcher::get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2)
{
    std::unordered_map<MapPoint *, Feature *> from_kf2_on_kf1;
    std::unordered_map<MapPoint *, Feature *> from_kf1_on_kf2;

    int out = 0;
    from_kf2_on_kf1 = this->match_frame_map_points(kf1, kf2->map_points);
    from_kf1_on_kf2 = this->match_frame_map_points(kf2, kf1->map_points);
    for (auto it = from_kf2_on_kf1.begin(); it != from_kf2_on_kf1.end(); it++)
    {
        if (from_kf1_on_kf2.find(it->first) != from_kf1_on_kf2.end())
            out++;
    }
    return out;
}

std::unordered_map<MapPoint *, Feature *> OrbMatcher::match_frame_reference_frame(KeyFrame *curr, KeyFrame *ref, ORBVocabulary *voc)
{
    DBoW2::FeatureVector curr_features = curr->features_vec;
    DBoW2::FeatureVector ref_features = ref->features_vec;
    std::unordered_map<MapPoint*, Feature*> out;

    DBoW2::FeatureVector::const_iterator f1it = ref_features.begin();
    DBoW2::FeatureVector::const_iterator f1end = ref_features.end();
    DBoW2::FeatureVector::const_iterator f2it = curr_features.begin();
    DBoW2::FeatureVector::const_iterator f2end = curr_features.end();
 
    int cate_ajung_acolo = 0;
    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (size_t i1 = 0, iend1 = f1it->second.size(); i1 < iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                MapPoint *mp_ref = ref->features[idx1].get_map_point();
                if (mp_ref == nullptr) continue;
                const cv::Mat &d1 = ref->orb_descriptors.row(idx1);

                int bestDist1 = 256;
                int bestIdx2 = -1;
                int bestDist2 = 256;

                for (size_t i2 = 0, iend2 = f2it->second.size(); i2 < iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];
                    MapPoint *mp_curr = curr->features[idx2].get_map_point();
                    if (mp_curr != nullptr) continue;
                    const cv::Mat &d2 = curr->orb_descriptors.row(idx2);
                    int dist = this->ComputeHammingDistance(d1, d2);
                    
                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx2 = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }
                if (bestDist1 > this->orb_descriptor_value) continue;
                if (bestDist1 > this->ration_first_second_match * bestDist2) continue;
                cate_ajung_acolo++;
                out.insert({mp_ref, &curr->features[bestIdx2]});
            }
            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = ref_features.lower_bound(f2it->first);
        }
        else
        {
            f2it = curr_features.lower_bound(f1it->first);
        }
    }
    std::cout << " cate ajung acolo " << cate_ajung_acolo << "\n";
    return out;
}