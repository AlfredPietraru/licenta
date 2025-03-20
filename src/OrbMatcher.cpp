#include "../include/OrbMatcher.h"

std::vector<std::pair<MapPoint*, Feature*>> OrbMatcher::match_two_consecutive_frames(KeyFrame *pref_kf, KeyFrame *curr_kf) {
    // for (int i = 0; i < 7; i++) {
    //     std::cout << pref_kf->Tiw.data()[i] << " "; 
    // }
    // std::cout << " prev_kf \n";
    // double angle_rad = -10.0 * M_PI / 180.0;
    // Eigen::Quaterniond help_quat = Eigen::Quaterniond(cos(angle_rad / 2), 0, 0, sin(angle_rad / 2));
    // Sophus::SE3d help_pose = Sophus::SE3d(help_quat, Eigen::Vector3d(0, 0, 0));
    // for (int i = 0; i < 7; i++) {
    //     std::cout << curr_kf->Tiw.data()[i] << " "; 
    // }
    // std::cout << " current_kf \n";
    // curr_kf->Tiw = curr_kf->Tiw * help_pose;
    // for (int i = 0; i < 7; i++) {
    //     std::cout << curr_kf->Tiw.data()[i] << " "; 
    // }
    // std::cout << " helped current \n";
    int MAX_NUMBER_ITERATIONS = 6;
    int WINDOW_INCREASE_FACTOR = 3;
    int ORB_DESCRIPTOR_INCREASE_FACTOR = 5;
    std::vector<std::pair<MapPoint*, Feature*>> out;
    std::cout << pref_kf->features.size() << " features size in prev\n";
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
        // AICI DE ADAUGAT LOGICA DE EXTINDERE PENTRU A GASI UN DESCRIPTOR VALID
        if (kp_idx == -1) {
            invalid_kp_idx++;
            continue;
        }
        out.push_back(std::pair<MapPoint*, Feature*>(mp, &curr_kf->features[kp_idx]));
    }
    std::cout << idx_mp_null << " " << invalid_kp_idx << " " << "problematic values\n";
    return out;
}