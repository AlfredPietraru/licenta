#include "../include/OrbMatcher.h"


// std::vector<MapPoint*> OrbMatcher::project_map_points_frame(std::vector<MapPoint*> map_points)  {

// } 
int inline::OrbMatcher::ComputeHammingDistance(const cv::Mat &desc1, const cv::Mat &desc2) {
    int distance = 0;
    for (int i = 0; i < desc1.cols; i++) {
        uchar v = desc1.at<uchar>(i) ^ desc2.at<uchar>(i); 
        distance += __builtin_popcount(v);
    }
    return distance;
}



std::unordered_map<MapPoint*, Feature*> OrbMatcher::match_two_consecutive_frames(KeyFrame *pref_kf, KeyFrame *curr_kf) {
    int MAX_NUMBER_ITERATIONS = 4;
    int WINDOW_INCREASE_FACTOR = 3;
    int ORB_DESCRIPTOR_INCREASE_FACTOR = 5;
    std::unordered_map<MapPoint*, Feature*> out;
    int idx_mp_null = 0;
    int invalid_kp_idx = 0;
    for (MapPoint *mp : pref_kf->map_points) {
        int kp_idx = mp->reproject_map_point(curr_kf, window, orb_descriptor_value);
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
            kp_idx = mp->reproject_map_point(curr_kf, current_window, current_orb_descriptor_value);
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



std::vector<MapPoint *> OrbMatcher::get_reprojected_map_points(KeyFrame *curr_frame, KeyFrame *reference_kf)
{
    std::vector<MapPoint *> out;
    for (MapPoint *mp : reference_kf->map_points) {
        if (mp->reproject_map_point(curr_frame, this->window, this->orb_descriptor_value) != -1) out.push_back(mp);
    }
    return out;
}

int OrbMatcher::get_number_common_mappoints_between_keyframes(KeyFrame *kf1, KeyFrame *kf2) {
    std::unordered_set<MapPoint*> common_map_points;
    int out = 0;
    for (MapPoint *mp : kf1->map_points) {
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

    for (MapPoint *mp : kf2->map_points) {
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