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
        cur_hamm_dist = this->ComputeHammingDistance(mp->orb_descriptor, kf->orb_descriptors.row(idx));
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

std::unordered_map<MapPoint*, Feature*> OrbMatcher::match_two_consecutive_frames(KeyFrame *pref_kf, KeyFrame *curr_kf) {
    int MAX_NUMBER_ITERATIONS = 4;
    int WINDOW_INCREASE_FACTOR = 3;
    int ORB_DESCRIPTOR_INCREASE_FACTOR = 5;
    std::unordered_map<MapPoint*, Feature*> out;
    int idx_mp_null = 0;
    int invalid_kp_idx = 0;
    for (MapPoint *mp : pref_kf->map_points) {
        int kp_idx = this->orb_matcher_reproject_map_point(curr_kf, mp);
        if (kp_idx != -1) {
            out.insert({mp, &curr_kf->features[kp_idx]});
            continue;
        }
        int current_window = window;
        int current_orb_descriptor_value = orb_descriptor_value;
        for (int i = 0; i < MAX_NUMBER_ITERATIONS; i++) {
            if (i % 2 == 0) {
                current_window += WINDOW_INCREASE_FACTOR;
            } else {
                current_orb_descriptor_value += ORB_DESCRIPTOR_INCREASE_FACTOR;
            }
            kp_idx = this->orb_matcher_reproject_map_point(curr_kf, mp);
            if (kp_idx != -1) break;
        }

        if (kp_idx == -1) {
            invalid_kp_idx++;
            continue;
        }
        out.insert({mp, &curr_kf->features[kp_idx]});
    }
    return out;
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
    cv::waitKey(0);
}


std::unordered_map<MapPoint*, Feature*> OrbMatcher::match_frame_map_points(KeyFrame* kf, std::vector<MapPoint*> map_points, bool keep_count) {
    std::unordered_map<MapPoint*, Feature*> out_values;
    for (MapPoint *mp : map_points) {
        Eigen::Vector3d point_camera_coordinates =  kf->fromWorldToImage(mp->wcoord);
        for (int i = 0; i < 3; i++) {
            if (point_camera_coordinates(i) < 0)  continue;
        }
        if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1) continue;
        if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1) continue;
        if (mp->dmax < point_camera_coordinates(2) || mp->dmin > point_camera_coordinates(2)) continue;
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        int min_hamm_dist = 10000;
        int cur_hamm_dist;
        int out_idx = -1;
        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, window);
        if (kps_idx.size() == 0) continue;
        for (int idx : kps_idx) {
            cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->orb_descriptors.row(idx));
            if (cur_hamm_dist < min_hamm_dist) {
                out_idx = idx;
                min_hamm_dist = cur_hamm_dist;
            }
        }
        if (min_hamm_dist > orb_descriptor_value || out_idx == -1 || kf->features[out_idx].get_map_point() != nullptr) continue;
        out_values.insert({mp, &kf->features[out_idx]});
        if (keep_count) {
            kf->currently_matched_points++;
            if (kf->currently_matched_points == kf->maximum_possible_map_points) break;
        }
    }
    // map point can be reprojected on image 
    return out_values;
}

std::unordered_map<MapPoint*, Feature*> OrbMatcher::match_frame_map_points(KeyFrame* kf, std::unordered_set<MapPoint*> map_points, bool keep_count) {
    std::unordered_map<MapPoint*, Feature*> out_values;
    for (MapPoint *mp : map_points) {
        Eigen::Vector3d point_camera_coordinates =  kf->fromWorldToImage(mp->wcoord);
        for (int i = 0; i < 3; i++) {
            if (point_camera_coordinates(i) < 0)  continue;
        }
        if (point_camera_coordinates(0) > kf->depth_matrix.cols - 1) continue;
        if (point_camera_coordinates(1) > kf->depth_matrix.rows - 1) continue;
        if (mp->dmax < point_camera_coordinates(2) || mp->dmin > point_camera_coordinates(2)) continue;
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        int min_hamm_dist = 10000;
        int cur_hamm_dist;
        int out_idx = -1;
        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, window);
        if (kps_idx.size() == 0) continue;
        for (int idx : kps_idx) {
            cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->orb_descriptors.row(idx));
            if (cur_hamm_dist < min_hamm_dist) {
                out_idx = idx;
                min_hamm_dist = cur_hamm_dist;
            }
        }
        if (min_hamm_dist > orb_descriptor_value || out_idx == -1 || kf->features[out_idx].get_map_point() != nullptr) continue;
        out_values.insert({mp, &kf->features[out_idx]});
        if (keep_count) {
            kf->currently_matched_points++;
            if (kf->currently_matched_points == kf->maximum_possible_map_points) break;
        }
    }
    // map point can be reprojected on image 
    return out_values;
}