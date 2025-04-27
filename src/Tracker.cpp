#include "../include/Tracker.h"

void compute_difference_between_positions(const Sophus::SE3d &estimated, const Sophus::SE3d &ground_truth, bool print_now)
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

    if (!print_now) return;
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
        delete this->prev_kf;
        this->prev_kf = this->current_kf;
        this->current_kf = new KeyFrame(this->prev_kf->Tcw, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 1, 
            frame, this->voc, this->reference_kf, this->prev_kf->reference_idx);
        return;
    }

    // de incercat ultima transformare ca estimare
    Sophus::SE3d pose_estimation = this->current_kf->Tcw;
    // Sophus::SE3d pose_estimation = this->current_kf->Tcw * this->prev_kf->Tcw.inverse() * this->current_kf->Tcw;
    this->prev_kf = this->current_kf;
    this->current_kf = new KeyFrame(pose_estimation, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 
        this->prev_kf->current_idx + 1, frame, this->voc, this->reference_kf, this->prev_kf->reference_idx);
}

Tracker::Tracker(Mat frame, Mat depth, Map *mapp, Sophus::SE3d pose, Config cfg, 
    ORBVocabulary* voc, Orb_Matcher orb_matcher_config) : mapp(mapp), voc(voc) {
    this->K = cfg.K;
    cv::cv2eigen(cfg.K, this->K_eigen);
    this->mDistCoef = cfg.distortion;
    this->fmf = new FeatureMatcherFinder(480, 640, cfg);
    this->motionOnlyBA = new MotionOnlyBA();
    this->matcher = new OrbMatcher(orb_matcher_config);
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_kps;
    cv::Mat descriptors;
    this->fmf->compute_keypoints_descriptors(frame, keypoints, undistorted_kps, descriptors);
    // this->current_kf = new KeyFrame(pose, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps,  descriptors, depth, 0, frame, this->voc);
    pose = Sophus::SE3d(Eigen::Matrix4d::Identity());
    // pose = Sophus::SE3d(Eigen::Quaterniond(-0.3986, 0.6132, 0.5962, -0.3311), Eigen::Vector3d(-0.6305, -1.3563, 1.6380));
    this->current_kf = new KeyFrame(pose, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps,  descriptors, depth, 0, frame, this->voc, nullptr, 0);
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

bool Tracker::Is_KeyFrame_needed(Map *mapp, int tracked_by_local_map)
{  
    float fraction = mapp->keyframes.size() <= 2 ? 0.75f : 0.4f;
    int nr_references = (int)mapp->keyframes.size() <= 2 ? 2 : 3;
    int points_seen_from_multiple_frames_reference = this->reference_kf->get_map_points_seen_from_multiple_frames(nr_references);
    bool weak_good_map_points_tracking = tracked_by_local_map <= 0.25 * points_seen_from_multiple_frames_reference;  
    bool needToInsertClose = this->current_kf->check_number_close_points();
    bool c1 = (this->current_kf->current_idx - this->reference_kf->current_idx) >= 30;
    bool c2 = weak_good_map_points_tracking || needToInsertClose; 
    bool c3 = ((tracked_by_local_map < points_seen_from_multiple_frames_reference * fraction) || needToInsertClose) && (tracked_by_local_map > 15);
    if ((c1 || c2) && c3) {
        std::cout << tracked_by_local_map << " atatea puncte urmarite in track local map\n";
        std::cout << points_seen_from_multiple_frames_reference << " atatea puncte urmarite din mai multe cadre\n";
        std::cout << "conditions that lead to that " << c1 << " " << weak_good_map_points_tracking << " " << needToInsertClose << " " << c3 << "\n";
    }
    return (c1 || c2) && c3;
}


void Tracker::TrackReferenceKeyFrame() {
    matcher->match_frame_reference_frame(this->current_kf, this->reference_kf);
    if (this->current_kf->mp_correlations.size() < 15) {
        std::cout << this->current_kf->mp_correlations.size() << " REFERENCE FRAME N A URMARIT SUFICIENTE MAP POINTS PENTRU OPTIMIZARE\n";
        exit(1);
    }
    this->current_kf->set_keyframe_position(this->motionOnlyBA->solve_ceres(this->current_kf));
    for (MapPoint *mp : this->current_kf->outliers) {
        bool removal_succesfull = Map::remove_map_point_from_keyframe(this->current_kf, mp);
        if (!removal_succesfull) {
            std::cout << "STERGEREA OUTLIER A ESUAT\n";
        }
    }
    if (this->current_kf->mp_correlations.size() < 15) {
        std::cout << this->current_kf->mp_correlations.size() << " REFERENCE FRAME NU A URMARIT SUFICIENTE INLIERE MAP POINTS\n";
        return;
    } 
}

void Tracker::TrackConsecutiveFrames() {
    int window = 15;
    matcher->match_consecutive_frames(this->current_kf, this->prev_kf, window);
    if (this->current_kf->mp_correlations.size() < 20) {
        matcher->match_consecutive_frames(this->current_kf, this->prev_kf, 2 * window);
        if (this->current_kf->mp_correlations.size() < 20) {
            std::cout << " \nURMARIREA INTRE FRAME-URI INAINTE DE OPTIMIZARE NU A FUNCTIONAT\n";
            return;
        }
    }
    this->current_kf->set_keyframe_position(this->motionOnlyBA->solve_ceres(this->current_kf));
    for (MapPoint *mp : this->current_kf->outliers) {
        Map::remove_map_point_from_keyframe(this->current_kf, mp);
    }
    if (this->current_kf->mp_correlations.size() < 10) {
        std::cout << " \nURMARIREA INTRE FRAME-URI NU A FUNCTIONAT DUPA OPTIMIZARE\n";
        return;
    } 
}

void Tracker::TrackLocalMap(Map *mapp) {
    mapp->track_local_map(this->current_kf, this->reference_kf);
    if (this->current_kf->mp_correlations.size() < 30) {
        std::cout << " \nPRREA PUTINE PUNCTE PROIECTATE DE LOCAL MAP INAINTE DE OPTIMIZARE\n";
        return;
    }
    this->current_kf->set_keyframe_position(this->motionOnlyBA->solve_ceres(this->current_kf));
    for (MapPoint *mp : this->current_kf->outliers) {
        bool removal_succesfull = Map::remove_map_point_from_keyframe(this->current_kf, mp);
        if (!removal_succesfull) {
            std::cout << "STERGEREA OUTLIER A ESUAT\n";
        }
    }
    int minim_number_points_necessary =  mapp->keyframes.size() <= 2 ? 30 : 50;
    if ((int)this->current_kf->mp_correlations.size() < minim_number_points_necessary) {
        std::cout << this->current_kf->mp_correlations.size() << " PREA PUTINE PUNCTE PROIECTATE CARE NU SUNT OUTLIERE DE CATRE LOCAL MAP\n";
        return;
    }
}

std::pair<KeyFrame*, bool> Tracker::tracking(Mat frame, Mat depth, Sophus::SE3d ground_truth_pose) {
    this->get_current_key_frame(frame, depth);
    auto start = high_resolution_clock::now();
    if (this->current_kf->current_idx - this->reference_kf->current_idx <= 2) {
        std::cout << "URMARIT CU AJUTORUL TRACK REFERENCE KEY FRAME\n" ;
        this->TrackReferenceKeyFrame();
    }
    if (this->current_kf->current_idx - this->reference_kf->current_idx > 2) {
        this->TrackConsecutiveFrames();
        if (this->current_kf->mp_correlations.size() < 20) {
            std::cout << "INTRE FRAME-URI NU A FUNCTIONAT TRACKING-ul\n";
            this->TrackReferenceKeyFrame();
        }
    } 
    auto end = high_resolution_clock::now();
    total_tracking_during_matching += duration_cast<milliseconds>(end - start).count();

    start = high_resolution_clock::now();
    this->TrackLocalMap(mapp);
    end = high_resolution_clock::now();
    total_tracking_during_local_map += duration_cast<milliseconds>(end - start).count();
    compute_difference_between_positions(this->current_kf->Tcw, ground_truth_pose, false);
    bool needed_keyframe = this->Is_KeyFrame_needed(mapp, this->current_kf->map_points.size()); 
    if (needed_keyframe) {
        this->reference_kf = this->current_kf;
        this->reference_kf->reference_idx += 1;
        this->prev_kf = this->current_kf;
        this->keyframes_from_last_global_relocalization = 0;
    }
    int wait_time = this->current_kf->current_idx < 295 ? 20 : 0;
    this->current_kf->debug_keyframe(frame, wait_time);
    return {this->current_kf, needed_keyframe};
}