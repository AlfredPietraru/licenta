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

std::unordered_map<MapPoint *, Feature *> OrbMatcher::checkOrientation(std::unordered_map<MapPoint *, Feature *> &correlation_current_frame,
                                                                       std::unordered_map<MapPoint *, Feature *> &correlation_prev_frame)
{
    std::vector<std::vector<MapPoint *>> histogram(30, std::vector<MapPoint *>());
    const float factor = 1.0f / 30;
    for (auto it = correlation_current_frame.begin(); it != correlation_current_frame.end(); it++)
    {
        MapPoint *mp = it->first;
        cv::KeyPoint current_kp = it->second->get_key_point();
        cv::KeyPoint prev_kp = correlation_prev_frame[mp]->get_key_point();
        float rot = prev_kp.angle - current_kp.angle;
        if (rot < 0.0) rot += 360.0f;
        int bin = lround(rot * factor);
        if (bin == 30) bin = 0;
        histogram[bin].push_back(mp);
    }
    int max = 0;
    int index = 0;
    for (int i = 0; i < 30; i++) {
        if (histogram[i].size() > max) {
            max = histogram[i].size();
            index = i;
        }
    }
    
    int threshold = 0.7 * max;
    std::vector<bool> keep(30, false);
    
    for (int i = 0; i < 30; i++) {
        if (histogram[i].size() > threshold || 
            (i == ((index+1)%30) || i == ((index+29)%30))) {
            keep[i] = true;
        }
    }
    for (int i = 0; i < 30; i++) {
        if (keep[i]) continue;
        for (auto mp : histogram[i]) {
            correlation_current_frame.erase(mp);
        }
    }
    return correlation_current_frame;
}

std::unordered_map<MapPoint *, Feature *> OrbMatcher::match_consecutive_frames(KeyFrame *kf, KeyFrame *prev_kf)
{
    std::unordered_map<MapPoint *, Feature *> out;
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    Eigen::Vector3d camera_to_map_view_ray;
    std::unordered_map<MapPoint *, Feature *> prev_frame_correlations = prev_kf->return_map_points_keypoint_correlation();

    Eigen::Vector3d kf_camer_center_prev_coordinates = kf->Tiw.rotationMatrix() * kf_camera_center + kf->Tiw.translation(); \
    const bool bForward  =   kf_camer_center_prev_coordinates(2) >  3.2;
    const bool bBackward = -kf_camer_center_prev_coordinates(2) > 3.2;

    for (auto it = prev_frame_correlations.begin(); it != prev_frame_correlations.end(); it++)
    {
        MapPoint *mp = it->first;
        if (mp == nullptr || kf->check_map_point_outlier(mp)) continue;
        Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < 0 || point_camera_coordinates(0) > kf->depth_matrix.cols - 1)
        continue;
        if (point_camera_coordinates(1) < 0 || point_camera_coordinates(1) > kf->depth_matrix.rows - 1)
        continue;
        if (point_camera_coordinates(2) < 1e-6) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;
        
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        
        
        int current_octave = it->second->get_key_point().octave;
        int scale_of_search = this->window * pow(1.2, current_octave);
        
        std::vector<int> kps_idx;
        if (bForward) {
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, current_octave - 1, 7);
        } else if (bBackward) {
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, 0, current_octave);
        } else {
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, current_octave - 1, current_octave + 1);
        }
        if (kps_idx.size() == 0) continue;
        
        int best_dist = 256;
        int best_idx = -1;
        
        for (int idx : kps_idx)
        {
            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            if (kf->features[idx].get_map_point() != nullptr) continue; // verifica sa fie asociat unui singur map point
            if (kf->features[idx].stereo_depth > 0)
            {
                double fake_rgbd = u - kf->K(0, 0) * 0.08 / point_camera_coordinates(2);
                float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
                if (er > scale_of_search) continue;
            }
            if (cur_hamm_dist < best_dist)
            {
                best_dist = cur_hamm_dist;
                best_idx = idx;
            }
        }
        if (best_dist > this->orb_descriptor_value) continue;
        out.insert({mp, &kf->features[best_idx]});
    }
    this->checkOrientation(out, prev_frame_correlations);
    return out;
}


int OrbMatcher::reproject_map_points(KeyFrame *kf, std::unordered_set<MapPoint *> map_points, int window_size) {
    int out = 0;
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    int points_which_can_be_found = 0;
    Eigen::Vector3d camera_to_map_view_ray;
    int out_points_found = 0;
    for (auto it = map_points.begin(); it != map_points.end(); it++)
    {
        MapPoint *mp = *it;
        if (kf->check_map_point_outlier(mp)) continue;
        Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < 0 || point_camera_coordinates(0) > kf->depth_matrix.cols - 1)
            continue;
        if (point_camera_coordinates(1) < 0 > kf->depth_matrix.rows - 1)
            continue;
        if (point_camera_coordinates(2) <= 1e-6) continue;
        
        double distance = (mp->wcoord_3d - kf_camera_center).norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;
        
        float radius = (dot_product >= 0.98) ? 2.5 : 4;
        radius *= window_size; 
        int predicted_scale = mp->predict_image_scale(distance);
        int scale_of_search = radius * pow(1.2, predicted_scale);
        
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        
        int lowest_dist = 256;
        int lowest_idx = -1;
        int lowest_level = -1;
        int second_lowest_dist = 256;
        int second_lowest_level = -1;
        int second_lowest_idx = -1;
        
        
        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, predicted_scale - 1, predicted_scale + 1);
        if (kps_idx.size() == 0)
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, 2 * scale_of_search, predicted_scale - 1, predicted_scale + 1);
        if (kps_idx.size() == 0) continue;
        
        for (int idx : kps_idx)
        {
            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            // if (kf->features[idx].get_map_point() != nullptr) continue; 
            if (kf->features[idx].stereo_depth < 1e-6) continue;
            
            double fake_rgbd = u - kf->K(0, 0) * 0.08 / point_camera_coordinates(2);
            float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
            if (er > scale_of_search) continue;
            
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
        
        if (lowest_dist > 50)  continue;
        if(lowest_level == second_lowest_level && lowest_dist > this->ratio_track_local_map * second_lowest_dist) continue;
        out++;
    }
    return out;
    
}
 
std::unordered_map<MapPoint *, Feature *> OrbMatcher::match_frame_map_points(KeyFrame *kf, std::unordered_set<MapPoint *> map_points, int window_size)
{
    std::unordered_map<MapPoint *, Feature *> out;
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    int points_which_can_be_found = 0;
    Eigen::Vector3d camera_to_map_view_ray;
    int out_points_found = 0;
    for (auto it = map_points.begin(); it != map_points.end(); it++)
    {
        MapPoint *mp = *it;
        if (kf->check_map_point_outlier(mp)) continue;
        Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < 0 || point_camera_coordinates(0) > kf->depth_matrix.cols - 1)
            continue;
        if (point_camera_coordinates(1) < 0 > kf->depth_matrix.rows - 1)
            continue;
        if (point_camera_coordinates(2) <= 1e-6) continue;
        
        double distance = (mp->wcoord_3d - kf_camera_center).norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;
        
        float radius = (dot_product >= 0.98) ? 2.5 : 4;
        radius *= window_size; 
        int predicted_scale = mp->predict_image_scale(distance);
        int scale_of_search = radius * pow(1.2, predicted_scale);
        
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        
        int lowest_dist = 256;
        int lowest_idx = -1;
        int lowest_level = -1;
        int second_lowest_dist = 256;
        int second_lowest_level = -1;
        int second_lowest_idx = -1;
        
        
        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, predicted_scale - 1, predicted_scale + 1);
        if (kps_idx.size() == 0)
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, 2 * scale_of_search, predicted_scale - 1, predicted_scale + 1);
        if (kps_idx.size() == 0) continue;
        out_points_found++;
        for (int idx : kps_idx)
        {
            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            if (kf->features[idx].get_map_point() != nullptr) continue; 
            if (kf->features[idx].stereo_depth < 1e-6) continue;
            
            double fake_rgbd = u - kf->K(0, 0) * 0.08 / point_camera_coordinates(2);
            float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
            if (er > scale_of_search) continue;

            
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
        if (lowest_dist > 50)  continue;
        if(lowest_level == second_lowest_level && lowest_dist > this->ratio_track_local_map * second_lowest_dist) continue;
        out.insert({mp, &kf->features[lowest_idx]});
    }
    return out;
}

int OrbMatcher::get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2)
{
    std::unordered_set<MapPoint *> map_points_f1 = kf1->map_points;
    std::unordered_set<MapPoint *> map_points_f2 = kf2->map_points;
    int out = 0;
    for (auto it = map_points_f1.begin(); it != map_points_f1.end(); it++)
    {
        if (map_points_f2.find(*it) != map_points_f2.end()) out++;
    }
    return out;
}

std::unordered_map<MapPoint *, Feature *> OrbMatcher::match_frame_reference_frame(KeyFrame *curr, KeyFrame *ref)
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
                int bestIdx = -1;
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
                        bestIdx = idx2;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }
                if (bestDist1 > 60) continue;
                if (bestDist1 > 0.7 * bestDist2) continue;
                cate_ajung_acolo++;
                out.insert({mp_ref, &curr->features[bestIdx]});
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
    std::unordered_map<MapPoint*, Feature*> out_ref = ref->return_map_points_keypoint_correlation();
    this->checkOrientation(out, out_ref);
    return out;
}