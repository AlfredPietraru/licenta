#include "../include/OrbMatcher.h"

int inline::OrbMatcher::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}


int OrbMatcher::orb_matcher_reproject_map_point(KeyFrame *kf, MapPoint *mp) {
    Eigen::Vector3d point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
    for (int i = 0; i < 3; i++) {
        if (point_camera_coordinates(i) < 0)  return -1;
    }
    if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1) return -1;
    if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1) return -1;
    if (mp->dmax < point_camera_coordinates(2) || mp->dmin > point_camera_coordinates(2)) return -1;
    // map point can be reprojected on image 
    double u = point_camera_coordinates(0);
    double v = point_camera_coordinates(1);
    int min_hamm_dist = 10000;
    int cur_hamm_dist;
    int out = -1;

    std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, window);
    if (kps_idx.size() == 0) return -1;
    for (int idx : kps_idx) {
        cur_hamm_dist = this->ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
        if (cur_hamm_dist < min_hamm_dist) {
            out = idx;
            min_hamm_dist = cur_hamm_dist;
        }
    }
    if (min_hamm_dist > orb_descriptor_value) return -1;
    return out; 
}


int OrbMatcher::get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2) {
    std::unordered_set<MapPoint*> common_map_points;
    int out = 0;
    for (MapPoint *mp : kf1->map_points) {
        if (kf2->map_points.find(mp) == kf2->map_points.end()) {
            if (this->orb_matcher_reproject_map_point(kf2, mp) != -1) {
                common_map_points.insert(mp);
                out += 1;
            }
        } else {
            out += 1;
            common_map_points.insert(mp);
        }
    }

    for (MapPoint *mp : kf2->map_points) {
        if (common_map_points.find(mp) != common_map_points.end()) continue;
        if (this->orb_matcher_reproject_map_point(kf1, mp) != -1) {
            common_map_points.insert(mp);
            out += 1;
        }
    }
    
    return out;
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


void OrbMatcher::debug_reprojection(std::unordered_set<MapPoint *>& local_map, std::unordered_map<MapPoint *, Feature*>& out_map, KeyFrame *first_kf, int window, int orb_descriptor_value) {
    std::vector<cv::KeyPoint> map_point_matched;
    for (auto it = out_map.begin(); it != out_map.end(); it++) {
        MapPoint *mp = it->first;
        int out = this->orb_matcher_reproject_map_point(first_kf, mp);
        if (out == -1) continue;
        map_point_matched.push_back(first_kf->features[out].get_key_point());
    }
    std::cout << local_map.size() << " dimensiune initiala local map\n";
    std::cout << out_map.size() << " map points proiectate pe imagine\n";
    std::cout <<  map_point_matched.size() << " mapp points cu orb descriptors matched\n";
    cv::Mat img2, img3, img4;
    cv::drawKeypoints(first_kf->frame, first_kf->get_all_keypoints(), img2, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT); //albastru
    cv::drawKeypoints(img2, map_point_matched, img3, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT); //verde
    // cv::drawKeypoints(img3, map_point_matched, img4, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT); // rosu
    cv::imshow("Display window", img3);
    cv::waitKey(100);
}

std::unordered_map<MapPoint*, Feature*> OrbMatcher::match_frame_map_points(KeyFrame* kf, std::unordered_set<MapPoint*> map_points) {
    std::unordered_map<MapPoint*, Feature*> out;
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    Eigen::Vector3d camera_to_map_view_ray; 
    for (MapPoint *mp : map_points) {
        Eigen::Vector3d point_camera_coordinates =  kf->fromWorldToImage(mp->wcoord);
        for (int i = 0; i < 3; i++) {
            if (point_camera_coordinates(i) < 0)  continue;
        }
        if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1) continue;
        if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1) continue;
        if (mp->dmax < point_camera_coordinates(2) || mp->dmin > point_camera_coordinates(2)) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;

        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        int min_hamm_dist = 10000;
        int cur_hamm_dist;
        int out_idx = -1;
        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, window);
        if (kps_idx.size() == 0) continue;
        for (int idx : kps_idx) {
            cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            if (cur_hamm_dist < min_hamm_dist) {
                out_idx = idx;
                min_hamm_dist = cur_hamm_dist;
            }
        }
        if (min_hamm_dist > orb_descriptor_value || out_idx == -1) continue;
        if (mp->is_safe_to_use) out.insert({mp, &kf->features[out_idx]});
    }
    return out;
}