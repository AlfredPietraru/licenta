#include "../include/OrbMatcher.h"

int inline::OrbMatcher::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}

void sort_values(std::vector<std::pair<MapPoint*, Feature*>>& map_points) {
    for (int i = 0; i < map_points.size() - 1; i++) {
        for (int j = i + 1; j < map_points.size(); j++) {
            if (map_points[i].second->depth > map_points[j].second->depth) {
            std::pair<MapPoint*, Feature*> current = map_points[i];
                map_points[i] = map_points[j];
                map_points[j] = current;
            } 
        }
    }
}

std::unordered_map<MapPoint*, Feature*> OrbMatcher::checkOrientation(std::unordered_map<MapPoint*, Feature*>& out) {
    return std::unordered_map<MapPoint*, Feature*>();
}

std::unordered_map<MapPoint*, Feature*> OrbMatcher::match_frame_map_points(KeyFrame* kf, std::unordered_set<MapPoint*> map_points) {
    std::unordered_map<MapPoint*, Feature*> out;
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    double window_size = 2.5;
    Eigen::Vector3d camera_to_map_view_ray; 
    for (MapPoint *mp : map_points) {
        Eigen::Vector3d point_camera_coordinates =  kf->fromWorldToImage(mp->wcoord);
        for (int i = 0; i < 3; i++) {
            if (point_camera_coordinates(i) < 0)  continue;
        }
        if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1) continue;
        if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1) continue;
        
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
        if (kps_idx.size() == 0) kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, 2 * this->window);

        for (int idx : kps_idx) {
            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            if (kf->features[idx].get_map_point() != nullptr) continue; // verifica sa fie asociat unui singur map point
            if (kf->features[idx].stereo_depth > 0) {
                float er = fabs(point_camera_coordinates(2) - kf->features[idx].stereo_depth);
                if (er > window_size * mp->predicted_scale) continue;
            }
            if (cur_hamm_dist < lowest_dist) {
                second_lowest_dist = lowest_dist;
                second_lowest_idx = lowest_idx;
                second_lowest_level = lowest_level;
                lowest_idx = idx;
                lowest_dist = cur_hamm_dist;
                lowest_level = kf->features[idx].kp.octave;
            } else if (cur_hamm_dist < second_lowest_dist) {
                second_lowest_dist = cur_hamm_dist;
                second_lowest_idx = idx;
                second_lowest_level = kf->features[idx].kp.octave;
            }
        }

        if (lowest_dist > this->orb_descriptor_value) continue;
        if (lowest_level == second_lowest_level && lowest_dist > this->ration_first_second_match * second_lowest_dist) continue;
        if (mp->is_safe_to_use) out.insert({mp, &kf->features[lowest_idx]});
        // also check if rotation is valid -> TO DO LATER;
    }
    // this->checkOrientation(out);
    return out;
}

int OrbMatcher::get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2) {
    std::unordered_map<MapPoint*, Feature*> from_kf2_on_kf1;
    std::unordered_map<MapPoint*, Feature*> from_kf1_on_kf2;

    int out = 0;
    from_kf2_on_kf1 = this->match_frame_map_points(kf1, kf2->map_points);
    from_kf1_on_kf2 = this->match_frame_map_points(kf2, kf1->map_points);
    for (auto it = from_kf2_on_kf1.begin(); it != from_kf2_on_kf1.end(); it++) {
        if (from_kf1_on_kf2.find(it->first) != from_kf1_on_kf2.end()) out++;
    }
    return out;
}


std::unordered_map<MapPoint*, Feature*> match_frame_reference_frame(KeyFrame *curr, KeyFrame *ref, ORBVocabulary *voc) {
    return std::unordered_map<MapPoint*, Feature*>();
}