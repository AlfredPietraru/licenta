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
    this->fmf->compute_keypoints_descriptors(frame, keypoints, undistorted_kps, descriptors);
    if (this->prev_kf == nullptr && this->current_kf != nullptr) {
        this->prev_kf = this->current_kf;
        this->current_kf = new KeyFrame(this->prev_kf->Tcw, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 1, frame, this->voc, this->reference_kf);
        return;
    }

    // de incercat ultima transformare ca estimare
    // Sophus::SE3d pose_estimation = this->current_kf->Tcw;
    Sophus::SE3d pose_estimation = this->current_kf->Tcw * this->prev_kf->Tcw.inverse() * this->current_kf->Tcw;
    this->prev_kf = this->current_kf;
    this->current_kf = new KeyFrame(pose_estimation, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 
        this->prev_kf->current_idx + 1, frame, this->voc, this->reference_kf);
}

Tracker::Tracker(Mat frame, Mat depth, Map *mapp, Sophus::SE3d pose, Config cfg, 
    ORBVocabulary* voc, Pnp_Ransac_Config pnp_ransac_cfg, Orb_Matcher orb_matcher_config) : voc(voc),  mapp(mapp) {
    this->K = cfg.K;
    cv::cv2eigen(cfg.K, this->K_eigen);
    this->mDistCoef = cfg.distortion;
    this->fmf = new FeatureMatcherFinder(480, 640, cfg);
    this->bundleAdjustment = new BundleAdjustment();
    this->matcher = new OrbMatcher(orb_matcher_config);
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_kps;
    cv::Mat descriptors;
    this->fmf->compute_keypoints_descriptors(frame, keypoints, undistorted_kps, descriptors);
    // this->current_kf = new KeyFrame(pose, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps,  descriptors, depth, 0, frame, this->voc);
    this->current_kf = new KeyFrame(Sophus::SE3d(Eigen::Matrix4d::Identity()), this->K_eigen, this->mDistCoef, keypoints, undistorted_kps,  descriptors, depth, 0, frame, this->voc, nullptr);
    this->reference_kf = this->current_kf;
    mapp->add_first_keyframe(this->reference_kf); 
    std::cout << "SFARSIT INITIALIZARE\n\n";
}



void Tracker::tracking_was_lost()
{
    std::cout << "TRACKING WAS LOSTT<< \n\n\n";
    std::cout << this->current_kf->current_idx << "\n";
    exit(1);
}

bool Tracker::Is_KeyFrame_needed(int tracked_by_local_map)
{
    bool needToInsertClose = this->current_kf->check_number_close_points();
    bool c1 = this->current_kf->current_idx - this->reference_kf->current_idx > 30;
    bool c2 = tracked_by_local_map < 0.25 * this->reference_kf->map_points.size() || needToInsertClose; 
    bool c3 = (this->current_kf->map_points.size() < this->reference_kf->map_points.size() * 0.9 || needToInsertClose) && tracked_by_local_map > 15;
    return (c1 || c2) && c3;
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
            dmatch.distance = OrbMatcher::ComputeHammingDistance(f->descriptor, f_ref->descriptor); 
            dmatches.push_back(dmatch);
    }
    std::vector<cv::DMatch> sub_dmatches;
    int i = 0;
    int j = 100;
    // copy(dmatches.begin() + i, dmatches.begin() + j, back_inserter(sub_dmatches));
    // cv::drawMatches(curr->frame, curr->get_all_keypoints(), ref->frame, ref->get_all_keypoints(), sub_dmatches, img_matches);
    // cv::imshow("Feature Matches", img_matches);
    // waitKey(0);
}


void Tracker::TrackReferenceKeyFrame(std::unordered_map<MapPoint *, Feature *>& matches) {
    matcher->match_frame_reference_frame(matches, this->current_kf, this->reference_kf);
    if (matches.size() < 15) {
        std::cout << matches.size() << " REFERENCE FRAME N A URMARIT SUFICIENTE MAP POINTS PENTRU OPTIMIZARE\n";
        exit(1);
    }
    this->current_kf->set_keyframe_position(this->bundleAdjustment->solve_ceres(this->current_kf, matches));
    for (MapPoint *mp : this->current_kf->outliers) {
        if (matches.find(mp) != matches.end()) {
            matches.erase(mp);
        }
    }
    if (matches.size() < 10) {
        std::cout << matches.size() << "REFERENCE FRAME NU A URMARIT SUFICIENTE INLIERE MAP POINTS\n";
        exit(1);
    } 
    for (auto it = matches.begin(); it != matches.end(); it++) {
        this->current_kf->add_map_point(it->first, it->second, it->first->orb_descriptor);
    }
}

void Tracker::TrackConsecutiveFrames(std::unordered_map<MapPoint *, Feature *>& matches) {
    int window = 15;
    matcher->match_consecutive_frames(matches, this->current_kf, this->prev_kf, window);
    if (matches.size() < 20) {
        matcher->match_consecutive_frames(matches, this->current_kf, this->prev_kf, 2 * window);
        if (matches.size() < 20) {
            std::cout << " \nURMARIREA INTRE FRAME-URI INAINTE DE OPTIMIZARE NU A FUNCTIONAT\n";
            return;
        }
    }
    this->current_kf->set_keyframe_position(this->bundleAdjustment->solve_ceres(this->current_kf, matches));
    for (MapPoint *mp : this->current_kf->outliers) {
        if (matches.find(mp) != matches.end()) {
            matches.erase(mp);
        }
    }
    if (matches.size() < 10) {
        std::cout << " \nURMARIREA INTRE FRAME-URI NU A FUNCTIONAT DUPA OPTIMIZARE\n";
        return;
    } 
    for (auto it = matches.begin(); it != matches.end(); it++) {
        this->current_kf->add_map_point(it->first, it->second, it->first->orb_descriptor);
    }
}

void Tracker::TrackLocalMap(std::unordered_map<MapPoint *, Feature *>& matches, Map *mapp) {
    matches.clear();
    mapp->track_local_map(matches, this->current_kf, this->reference_kf);
    if (matches.size() < 30) {
        std::cout << " \nPRREA PUTINE PUNCTE PROIECTATE DE LOCAL MAP INAINTE DE OPTIMIZARE\n";
        return;
    }
    this->current_kf->set_keyframe_position(this->bundleAdjustment->solve_ceres(this->current_kf, matches));
    for (MapPoint *mp : this->current_kf->outliers) {
        if (matches.find(mp) != matches.end()) {
            matches.erase(mp);
        }
    }
    if (matches.size() < 30) {
        std::cout << matches.size() << " PREA PUTINE PUNCTE PROIECTATE CARE NU SUNT OUTLIERE DE CATRE LOCAL MAP\n";
        return;
    }
    for (auto it = matches.begin(); it != matches.end(); it++) {
        this->current_kf->add_map_point(it->first, it->second, it->first->orb_descriptor);
    }
}

std::pair<KeyFrame*, bool> Tracker::tracking(Mat frame, Mat depth, Sophus::SE3d ground_truth_pose) {
    this->get_current_key_frame(frame, depth);
    std::unordered_map<MapPoint *, Feature *> matches;
    // std::cout << this->current_kf->Tcw.matrix() << "\n\n";
    if (this->current_kf->current_idx - this->reference_kf->current_idx <= 2) {
        std::cout << "URMARIT CU AJUTORUL TRACK REFERENCE KEY FRAME\n" ;
        this->TrackReferenceKeyFrame(matches);
    }
    if (this->current_kf->current_idx - this->reference_kf->current_idx > 2) {
        this->TrackConsecutiveFrames(matches);
        if (matches.size() < 20) {
            matches.clear();
            std::cout << "INTRE FRAME-URI NU A FUNCTIONAT TRACKING-ul\n";
            this->TrackReferenceKeyFrame(matches);
        }
    } 

    // std::cout << this->current_kf->Tcw.matrix() << "\n\n";
    this->TrackLocalMap(matches, mapp);
    // std::cout << this->current_kf->Tcw.matrix() << "\n\n";
    // compute_difference_between_positions(this->current_kf->Tcw, ground_truth_pose);
    // int wait_time = 30 ? this->current_kf->current_idx < 173 : 0;
    int wait_time = 20;
    this->current_kf->debug_keyframe(frame, wait_time, matches, matches);
    bool needed_keyframe = this->Is_KeyFrame_needed(matches.size()); 
    if (needed_keyframe) {
        std::cout << "UN KEYFRAME TREBUIE ADAUGAT\n\n\n";
        this->reference_kf = this->current_kf;
        this->prev_kf = this->current_kf;
        this->keyframes_from_last_global_relocalization = 0;
    }
    std::cout << this->current_kf->mp_correlations.size() << " map point correlate cu un feature\n";
    return {this->current_kf, needed_keyframe};
}