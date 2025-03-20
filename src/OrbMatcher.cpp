#include "../include/OrbMatcher.h"

std::vector<std::pair<MapPoint*, Feature*>> OrbMatcher::match_two_consecutive_frames(KeyFrame *pref_kf, KeyFrame *curr_kf) {
    int MAX_NUMBER_ITERATIONS = 6;
    int WINDOW_INCREASE_FACTOR = 3;
    int ORB_DESCRIPTOR_INCREASE_FACTOR = 5;
    std::vector<std::pair<MapPoint*, Feature*>> out;
    int idx_mp_null = 0;
    int invalid_kp_idx = 0;
    for (Feature f : pref_kf->features) {
        MapPoint *mp = f.get_map_point();
        if (mp == nullptr)  {
            idx_mp_null++;
            continue;
        }
        int kp_idx = mp->reproject_map_point(curr_kf, window, orb_descriptor_value);
        int current_window = window;
        int current_orb_descriptor_value = orb_descriptor_value;
        for (int i = 0; i < MAX_NUMBER_ITERATIONS; i++) {
            if (i % 2 == 0) {
                current_window += WINDOW_INCREASE_FACTOR;
            } else {
                current_orb_descriptor_value += ORB_DESCRIPTOR_INCREASE_FACTOR;
            }
            kp_idx = mp->reproject_map_point(curr_kf, current_window, current_orb_descriptor_value);
            if (kp_idx != -1) break;
        }
        if (kp_idx == -1) {
            invalid_kp_idx++;
            continue;
        }
        out.push_back(std::pair<MapPoint*, Feature*>(mp, &curr_kf->features[kp_idx]));
    }
    // std::cout << idx_mp_null << " " << invalid_kp_idx << " " << "problematic values\n";
    return out;
}

std::vector<MapPoint *> OrbMatcher::get_reprojected_map_points(KeyFrame *curr_frame, KeyFrame *reference_kf)
{
    std::vector<MapPoint *> out;
    for (int i = 0; i < reference_kf->features.size(); i++) {
        MapPoint *mp = reference_kf->features[i].get_map_point();
        if (mp == nullptr) continue;
        if (mp->reproject_map_point(curr_frame, this->window, this->orb_descriptor_value) != -1) out.push_back(mp);
    }
    return out;
}

int OrbMatcher::get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2) {
    std::unordered_set<MapPoint*> common_map_points;
    int out = 0;
    for (int i = 0; i < kf1->features.size(); i++) {
        MapPoint *mp = kf1->features[i].get_map_point();
        if (mp == nullptr) continue;
        if (kf2->map_points.find(mp) == kf2->map_points.end()) {
            if (mp->reproject_map_point(kf2, this->window, this->orb_descriptor_value) != -1) {
                common_map_points.insert(mp);
                out += 1;
            }
        } else {
            out += 1;
            common_map_points.insert(mp);
        }
    }

    for (int i = 0; i < kf2->features.size(); i++) {
        MapPoint *mp = kf2->features[i].get_map_point();
        if (mp == nullptr) continue;
        if (common_map_points.find(mp) != common_map_points.end()) continue;
        if (mp->reproject_map_point(kf1, this->window, this->orb_descriptor_value) != -1) {
            common_map_points.insert(mp);
            out += 1;
        }
    }
    return out;
}


void OrbMatcher::debug_reprojection(std::unordered_set<MapPoint *>& local_map, std::unordered_map<MapPoint *, Feature*>& out_map, KeyFrame *first_kf, int window, int orb_descriptor_value) {
    std::vector<cv::KeyPoint> map_point_matched;
    for (auto it = out_map.begin(); it != out_map.end(); it++) {
        MapPoint *mp = it->first;
        int out = mp->reproject_map_point(first_kf, window, orb_descriptor_value);
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