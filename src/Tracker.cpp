#include "../include/Tracker.h"


void compute_difference_between_positions(const Sophus::SE3d &estimated, const Sophus::SE3d &ground_truth)
{
    
    Sophus::SE3d relative = ground_truth.inverse() * estimated;
    double APE = relative.log().norm();

    Eigen::Vector3d angle_axis = relative.so3().log();
    angle_axis = angle_axis * 180 / M_PI;

    Eigen::Quaterniond q_rel = relative.unit_quaternion();
    
    double total_angle_diff = 2.0 * std::acos(q_rel.w()) * 180.0 / M_PI;
    if (total_angle_diff > 180.0) {
        total_angle_diff = 360.0 - total_angle_diff;
    }
    
    Eigen::Vector3d t_rel = relative.translation();
    double translation_diff = t_rel.norm();

    std::cout << "APE score: " << APE << "\n";
    std::cout << "Rotation Difference (Total): " << total_angle_diff << " degrees\n";
    std::cout << "Rotation Difference (X): " << angle_axis(0) << " degrees\n";
    std::cout << "Rotation Difference (Y): " << angle_axis(1) << " degrees\n";
    std::cout << "Rotation Difference (Z): " << angle_axis(2) << " degrees\n";
    std::cout << "Translation Difference: " << translation_diff << " meters\n\n";
}


void print_pose(Sophus::SE3d pose, std::string message) {
    for (int i = 0; i < 7; i++)
    {
        std::cout << pose.data()[i] << " ";
    }
    std::cout << message << "\n";
}

void Tracker::get_current_key_frame(Mat frame, Mat depth) {
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_kps;
    cv::Mat descriptors;
    std::pair<std::pair<std::vector<cv::KeyPoint>, cv::Mat>, std::vector<cv::KeyPoint>> out =  this->fmf->compute_keypoints_descriptors(frame);
    keypoints = out.first.first;
    descriptors = out.first.second;
    undistorted_kps = out.second;
    if (this->prev_kf == nullptr && this->current_kf != nullptr) {
        this->prev_kf = this->current_kf;
        this->current_kf = new KeyFrame(this->prev_kf->Tiw, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 1, frame, this->voc);
        return;
    }

    // de incercat ultima transformare ca estimare
    // Sophus::SE3d pose_estimation = this->current_kf->Tiw;
    Sophus::SE3d pose_estimation = this->current_kf->Tiw * this->prev_kf->Tiw.inverse() * this->current_kf->Tiw;
    this->prev_kf = this->current_kf;
    this->current_kf = new KeyFrame(pose_estimation, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 
        this->prev_kf->current_idx + 1, frame, this->voc);
}

void Tracker::initialize(Mat frame, Mat depth, Map* mapp, Sophus::SE3d pose)
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_kps;
    cv::Mat descriptors;
    std::pair<std::pair<std::vector<cv::KeyPoint>, cv::Mat>, std::vector<cv::KeyPoint>> out =  this->fmf->compute_keypoints_descriptors(frame);
    keypoints = out.first.first;
    descriptors = out.first.second;
    undistorted_kps = out.second;
    this->current_kf = new KeyFrame(pose, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps,  descriptors, depth, 0, frame, this->voc);
    // this->current_kf = new KeyFrame(Sophus::SE3d(Eigen::Matrix4d::Identity()), this->K_eigen, this->mDistCoef, keypoints, undistorted_kps,  descriptors, depth, 0, frame, this->voc);

    this->reference_kf = this->current_kf;
    mapp->add_first_keyframe(this->reference_kf);
    // this->mapDrawer = new MapDrawer(mapp, pose.matrix()); 
    std::cout << "SFARSIT INITIALIZARE\n\n";
}

void Tracker::tracking_was_lost()
{
    std::cout << "TRACKING WAS LOSTT<< \n\n\n";
    std::cout << this->current_kf->current_idx << "\n";
    exit(1);
}

bool Tracker::Is_KeyFrame_needed()
{
    this->keyframes_from_last_global_relocalization += 1;
    bool c1 = this->keyframes_from_last_global_relocalization > 20;
    bool c2 = this->current_kf->current_idx - this->reference_kf->current_idx > 20;
    bool c3 = this->current_kf->check_number_close_points() < 100;
    bool c4 = reference_kf->map_points.size() / 10 > this->current_kf->check_number_close_points();
    return c1 && c2 && c3 && c4 && true;
}


void check_feature_matching(KeyFrame *curr, KeyFrame *ref, std::unordered_map<MapPoint *, Feature *>& matches) {
    std::unordered_map<MapPoint *, Feature *> ref_map_feature = ref->mp_correlations;
    cv::Mat img_matches;
    std::vector<cv::DMatch> dmatches;
    for (const auto& match : matches) {
        MapPoint* mp = match.first;
        Feature *f = match.second;

        if (ref_map_feature.find(mp) == ref_map_feature.end()) continue;
        Feature *f_ref = ref_map_feature[mp];
            cv::DMatch dmatch;
            dmatch.queryIdx = f->idx; 
            dmatch.trainIdx = f_ref->idx;  
            dmatch.distance = f->ComputeHammingDistance(f->descriptor, f_ref->descriptor); 
            dmatches.push_back(dmatch);
    }
    std::vector<cv::DMatch> sub_dmatches;
    int i = 0;
    int j = 100;
    copy(dmatches.begin() + i, dmatches.begin() + j, back_inserter(sub_dmatches));
    cv::drawMatches(curr->frame, curr->get_all_keypoints(), ref->frame, ref->get_all_keypoints(), sub_dmatches, img_matches);
    cv::imshow("Feature Matches", img_matches);
    waitKey(0);
}


std::unordered_map<MapPoint *, Feature *> Tracker::TrackReferenceKeyFrame() {
    std::unordered_map<MapPoint *, Feature *> matches = matcher->match_frame_reference_frame(this->current_kf, this->reference_kf);
    if (matches.size() < 15) {
        std::cout << matches.size() << "REFERENCE FRAME N A URMARIT SUFICIENTE MAP POINTS PENTRU OPTIMIZARE\n";
        std::cout << this->current_kf->current_idx << "\n";
        exit(1);
    }
    this->current_kf->Tiw = this->bundleAdjustment->solve_ceres(this->current_kf, matches);
    std::unordered_map<MapPoint *, Feature*> filtered_matches;
    for (auto it = matches.begin(); it != matches.end(); it++) {
        if (this->current_kf->check_map_point_outlier(it->first)) continue;
        filtered_matches.insert({it->first, it->second});
    }
    if (filtered_matches.size() < 10) {
        std::cout << filtered_matches.size() << "REFERENCE FRAME NU A URMARIT SUFICIENTE INLIERE MAP POINTS\n";
        std::cout << this->current_kf->current_idx << "\n";
        exit(1);
    }
    return filtered_matches; 
}

std::unordered_map<MapPoint*, Feature*> Tracker::TrackConsecutiveFrames() {
    int window = 15;
    std::unordered_map<MapPoint*, Feature*> matches = matcher->match_consecutive_frames(this->current_kf, this->prev_kf, window);
    if (matches.size() < 20) {
        matches = matcher->match_consecutive_frames(this->current_kf, this->prev_kf, 2 * window);
        if (matches.size() < 20) {
            std::cout << "\nURMARIREA INTRE FRAME-URI INAINTE DE OPTIMIZARE NU A FUNCTIONAT\n";
            exit(1);
        }
    }
    this->current_kf->Tiw = this->bundleAdjustment->solve_ceres(this->current_kf, matches);
    std::unordered_map<MapPoint *, Feature*> filtered_matches;
    for (auto it = matches.begin(); it != matches.end(); it++) {
        if (this->current_kf->check_map_point_outlier(it->first)) continue;
        filtered_matches.insert({it->first, it->second});
    }

    if (filtered_matches.size() < 10) {
        std::cout << "\nURMARIREA INTRE FRAME-URI NU A FUNCTIONAT DUPA OPTIMIZARE\n";
        return filtered_matches;
    }
    return filtered_matches; 
}

// de verificat local map bine de tot
std::unordered_map<MapPoint*, Feature*> Tracker::TrackLocalMap(Map *mapp) {
    std::unordered_map<MapPoint *, Feature *> new_matches = mapp->track_local_map(this->current_kf, this->reference_kf);
    if (new_matches.size() < 30) {
        std::cout << "\nPRREA PUTINE PUNCTE PROIECTATE DE LOCAL MAP INAINTE DE OPTIMIZARE\n";
        exit(1);
    }
    this->current_kf->Tiw = this->bundleAdjustment->solve_ceres(this->current_kf, new_matches);
    int out = 0; 
    for (auto it = new_matches.begin(); it != new_matches.end(); it++) {
        if (this->current_kf->check_map_point_outlier(it->first)) continue;
        out += 1;
    }
    if (out < 30) {
        std::cout << out << " PREA PUTINE PUNCTE PROIECTATE CARE NU SUNT OUTLIERE DE CATRE LOCAL MAP\n";
        std::cout << this->current_kf->current_idx << " ATATEA FRAME-URI URMARITE\n";
        exit(1);
    }
    return new_matches;
}

std::pair<KeyFrame*, bool> Tracker::tracking(Mat frame, Mat depth, Map *mapp, Sophus::SE3d ground_truth_pose) {
    this->get_current_key_frame(frame, depth);
    std::unordered_map<MapPoint *, Feature *> matches;
    if (this->current_kf->current_idx - this->reference_kf->current_idx <= 2) {
        std::cout << "inca urmarit cu ajutorul track reference frame\n";
        matches = this->TrackReferenceKeyFrame();
    } else {
        matches = this->TrackConsecutiveFrames();
        if (matches.size() < 20) {
            std::cout << "intre frame-uri nu s-a gasit suficient\n";
            matches = this->TrackReferenceKeyFrame();
        }
    } 
    this->current_kf->correlate_map_points_to_features_current_frame(matches);
    // compute_difference_between_positions(this->current_kf->Tiw, ground_truth_pose);
    std::unordered_map<MapPoint *, Feature *>  new_matches = this->TrackLocalMap(mapp);
    this->current_kf->correlate_map_points_to_features_current_frame(new_matches);
    compute_difference_between_positions(this->current_kf->Tiw, ground_truth_pose);
    // this->current_kf->debug_keyframe(50, matches, new_matches);
    bool needed_keyframe = this->Is_KeyFrame_needed(); 
    if (needed_keyframe) {
        std::cout << "UN KEYFRAME TREBUIE ADAUGAT\n\n\n";
        this->reference_kf = this->current_kf;
        this->prev_kf = this->current_kf;
        this->keyframes_from_last_global_relocalization = 0;
    }
    std::cout << this->current_kf->map_points.size() << " map point cu care ramane in final\n";
    std::cout << this->current_kf->mp_correlations.size() << " map point correlate cu un feature\n";
    return {this->current_kf, needed_keyframe};
}