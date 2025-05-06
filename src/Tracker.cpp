#include "../include/Tracker.h"


void compute_difference_between_positions(const Sophus::SE3d &estimated, const Sophus::SE3d &ground_truth, bool print_now)
{
    if (!print_now)
        return;
    Sophus::SE3d relative = ground_truth.inverse() * estimated;
    double APE = relative.log().norm();

    Eigen::Vector3d angle_axis = relative.so3().log();
    angle_axis = angle_axis * 180 / M_PI;

    Eigen::Quaterniond q_rel = relative.unit_quaternion();

    double total_angle_diff = 2.0 * std::acos(q_rel.w()) * 180.0 / M_PI;
    if (total_angle_diff > 180.0)
    {
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

void print_pose(Sophus::SE3d pose, std::string message)
{
    for (int i = 0; i < 7; i++)
    {
        std::cout << pose.data()[i] << " ";
    }
    std::cout << message << "\n";
}

void Tracker::get_current_key_frame(Mat frame, Mat depth)
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_kps;
    cv::Mat descriptors;
    this->fmf->compute_keypoints_descriptors(frame, keypoints, undistorted_kps, descriptors);
    // primul cadru
    if (this->prev_kf == nullptr && this->current_kf == nullptr)
    {
        Sophus::SE3d pose = Sophus::SE3d(Eigen::Matrix4d::Identity());
        this->current_kf = new KeyFrame(pose, this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 0, frame, this->voc);
        this->velocity = pose;
        return;
    }
    // al doilea cadru
    if (this->prev_kf == nullptr && this->current_kf != nullptr)
    {
        this->velocity = this->current_kf->Tcw;
        this->prev_kf = this->current_kf;
        this->current_kf = new KeyFrame(this->prev_kf, keypoints, undistorted_kps, descriptors, depth);
        return;
    }

    // restul cadrelor
    this->velocity = this->current_kf->Tcw * this->prev_kf->Tcw.inverse();
    if (!this->prev_kf->isKeyFrame)
    {
        delete this->prev_kf;
        this->prev_kf = nullptr;
    }
    this->prev_kf = this->current_kf;
    this->current_kf = new KeyFrame(this->prev_kf, keypoints, undistorted_kps, descriptors, depth);
}

Tracker::Tracker(Map *mapp, Config cfg, ORBVocabulary *voc, Orb_Matcher orb_matcher_config) : mapp(mapp), voc(voc)
{
    this->K = cfg.K;
    cv::cv2eigen(cfg.K, this->K_eigen);
    this->mDistCoef = cfg.distortion;
    this->fmf = new FeatureMatcherFinder(480, 640, cfg);
    this->motionOnlyBA = new MotionOnlyBA();
    this->matcher = new OrbMatcher(orb_matcher_config);
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
    const bool at_the_beginning = mapp->keyframes.size() <= 2;
    float fraction = at_the_beginning * 0.75f + 0.4f * (!at_the_beginning);
    int nr_references = at_the_beginning * 2 + 3 * (!at_the_beginning);
    const int nr_points_matched = (int)current_kf->mp_correlations.size();
    const int points_seen_from_multiple_frames_reference = this->reference_kf->get_map_points_seen_from_multiple_frames(nr_references);
    const bool weak_good_map_points_tracking = nr_points_matched <= 0.25 * points_seen_from_multiple_frames_reference;
    const bool needToInsertClose = this->current_kf->check_number_close_points();
    const bool c1 = (this->current_kf->current_idx - this->reference_kf->current_idx) >= 30;
    const bool c2 = weak_good_map_points_tracking || needToInsertClose;
    const bool c3 = ((nr_points_matched < points_seen_from_multiple_frames_reference * fraction) || needToInsertClose) && (nr_points_matched > 15);
    
    if ((c1 || c2) && c3)
    {
        std::cout << nr_points_matched << " atatea puncte urmarite in track local map\n";
        std::cout << points_seen_from_multiple_frames_reference << " atatea puncte urmarite din mai multe cadre\n";
        std::cout << "conditions that lead to that " << c1 << " " << weak_good_map_points_tracking << " " << needToInsertClose << " " << c3 << "\n\n";
    }
    return (c1 || c2) && c3;
}

void Tracker::TrackReferenceKeyFrame()
{
    if (this->prev_kf != nullptr)
    {
        this->current_kf->set_keyframe_position(this->velocity * this->prev_kf->Tcw);
    }
    auto start = high_resolution_clock::now();
    matcher->match_frame_reference_frame(this->current_kf, this->reference_kf);
    auto end = high_resolution_clock::now();
    this->orb_matching_time += duration_cast<milliseconds>(end - start).count();

    if (this->current_kf->mp_correlations.size() < 15)
    {
        std::cout << this->current_kf->mp_correlations.size() << " REFERENCE FRAME N A URMARIT SUFICIENTE MAP POINTS PENTRU OPTIMIZARE\n";
        exit(1);
    }
    start = high_resolution_clock::now();
    this->motionOnlyBA->solve_ceres(this->current_kf, false);
    end = high_resolution_clock::now();
    this->motion_only_ba_time += duration_cast<milliseconds>(end - start).count();
    if (this->current_kf->mp_correlations.size() < 10)
    {
        std::cout << this->current_kf->mp_correlations.size() << " REFERENCE FRAME NU A URMARIT SUFICIENTE INLIERE MAP POINTS\n";
        return;
    }
}

void Tracker::TrackConsecutiveFrames()
{
    int window = 15;
    this->current_kf->set_keyframe_position(this->velocity * this->prev_kf->Tcw);
    auto start = high_resolution_clock::now();
    matcher->match_consecutive_frames(this->current_kf, this->prev_kf, window);
    auto end = high_resolution_clock::now();
    this->orb_matching_time += duration_cast<milliseconds>(end - start).count();
    if (this->current_kf->mp_correlations.size() < 20)
    {
        matcher->match_consecutive_frames(this->current_kf, this->prev_kf, 2 * window);
        if (this->current_kf->mp_correlations.size() < 20)
        {
            std::cout << this->current_kf->mp_correlations.size() << "  URMARIREA INTRE FRAME-URI INAINTE DE OPTIMIZARE NU A FUNCTIONAT\n";
            return;
        }
    }
    start = high_resolution_clock::now();
    this->motionOnlyBA->solve_ceres(this->current_kf, false);
    end = high_resolution_clock::now();
    this->motion_only_ba_time += duration_cast<milliseconds>(end - start).count();
    if (this->current_kf->mp_correlations.size() < 10)
    {
        std::cout << " \nURMARIREA INTRE FRAME-URI NU A FUNCTIONAT DUPA OPTIMIZARE\n";
        return;
    }
}

KeyFrame *Tracker::FindReferenceKeyFrame()
{
    this->local_keyframes.clear();
    unordered_map<KeyFrame *, int> umap = this->mapp->get_keyframes_connected(this->current_kf, 0);
    KeyFrame *new_ref_kf = nullptr;
    int maxim_nr_map_points = -1;
    for (auto it = umap.begin(); it != umap.end(); it++)
    {
        this->local_keyframes.insert(it->first);
        if (it->second > maxim_nr_map_points)
        {
            new_ref_kf = it->first;
            maxim_nr_map_points = it->second;
        }
    }
    return new_ref_kf;
}

void Tracker::TrackLocalMap(Map *mapp)
{
    KeyFrame *current_ref_kf = this->FindReferenceKeyFrame();
    auto start = high_resolution_clock::now();
    mapp->track_local_map(this->current_kf, current_ref_kf, this->local_keyframes);
    auto end = high_resolution_clock::now();
    this->orb_matching_time += duration_cast<milliseconds>(end - start).count();
    if (this->current_kf->mp_correlations.size() < 30)
    {
        std::cout << " \nPRREA PUTINE PUNCTE PROIECTATE DE LOCAL MAP INAINTE DE OPTIMIZARE\n";
        return;
    }
    start = high_resolution_clock::now();
    this->motionOnlyBA->solve_ceres(this->current_kf, false);
    end = high_resolution_clock::now();
    this->motion_only_ba_time += duration_cast<milliseconds>(end - start).count();
    int minim_number_points_necessary = mapp->keyframes.size() <= 2 ? 30 : 50;
    if ((int)this->current_kf->mp_correlations.size() < minim_number_points_necessary)
    {
        std::cout << this->current_kf->mp_correlations.size() << " PREA PUTINE PUNCTE PROIECTATE CARE NU SUNT OUTLIERE DE CATRE LOCAL MAP\n";
        return;
    }
}

KeyFrame *Tracker::tracking(Mat frame, Mat depth, Sophus::SE3d ground_truth_pose)
{
    auto start = high_resolution_clock::now();
    this->get_current_key_frame(frame, depth);
    if (this->is_first_keyframe)
    {
        this->is_first_keyframe = false;
        this->current_kf->isKeyFrame = true;
        this->reference_kf = this->current_kf;
        std::cout << "PRIMUL KEYFRAME VA FI ADAUGAT\n";
        return this->current_kf;
    }

    if (this->current_kf->current_idx - this->reference_kf->current_idx <= 2)
    {
        std::cout << "URMARIT CU AJUTORUL TRACK REFERENCE KEY FRAME\n";
        this->TrackReferenceKeyFrame();
    }
    if (this->current_kf->current_idx - this->reference_kf->current_idx > 2)
    {
        this->TrackConsecutiveFrames();
        if (this->current_kf->mp_correlations.size() < 20)
        {
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
    this->current_kf->isKeyFrame = this->Is_KeyFrame_needed();

    if (this->current_kf->isKeyFrame)
    {
        this->reference_kf = this->current_kf;
    }

    this->current_kf->reference_kf = this->reference_kf;
    // int wait_time = this->current_kf->current_idx < 225 ? 0 : 0;
    // this->current_kf->debug_keyframe(frame, wait_time);
    return this->current_kf;
}