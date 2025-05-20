#include "../include/Tracker.h"


Tracker::Tracker(Map *mapp, Config cfg, ORBVocabulary *voc) : mapp(mapp), voc(voc)
{
    this->K = cfg.K;
    cv::cv2eigen(cfg.K, this->K_eigen);
    this->mDistCoef = cfg.distortion;
    this->motionOnlyBA = new MotionOnlyBA();
    this->matcher = new OrbMatcher(cfg.orb_matcher);
    std::cout << "SFARSIT INITIALIZARE\n\n";
}

Sophus::SE3d Tracker::GetVelocityNextFrame() {
    if (frames_seen < 2) return Sophus::SE3d(Eigen::Matrix4d::Identity());
    // return Sophus::SE3d(Eigen::Matrix4d::Identity());
    if (this->current_kf->current_idx - this->reference_kf->current_idx <= 2) 
        return Sophus::SE3d(Eigen::Matrix4d::Identity());
    return this->prev_kf->Tcw * this->prev_prev_kf->Tcw.inverse();
}

void Tracker::UndistortKeyPoints(std::vector<cv::KeyPoint>& kps, std::vector<cv::KeyPoint>& u_kps)
{
    if(mDistCoef[0]==0.0) {
        std::copy(kps.begin(), kps.end(), std::back_inserter(u_kps));
        return;
    }
    cv::Mat mat(kps.size(), 2, CV_32F);
    for(long unsigned int i = 0; i < kps.size(); i++)
    {
        mat.at<float>(i,0)=kps[i].pt.x;
        mat.at<float>(i,1)=kps[i].pt.y;
    }

    mat=mat.reshape(2);
    cv::undistortPoints(mat, mat, this->K, mDistCoef,cv::Mat(),this->K);
    mat=mat.reshape(1);
    
    for(long unsigned int i = 0; i < kps.size(); i++)
    {
        cv::KeyPoint kp = kps[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        u_kps.push_back(kp);
    }
}

void Tracker::GetNextFrame(Mat frame, Mat depth)
{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> undistorted_kps;
    cv::Mat descriptors;
    // this->fmf->compute_keypoints_descriptors(frame, keypoints, undistorted_kps, descriptors);
    (*this->extractor)(frame, keypoints, descriptors);
    this->UndistortKeyPoints(keypoints, undistorted_kps);
    switch (frames_seen) {
        case 0:
            this->current_kf = new KeyFrame(Sophus::SE3d(Eigen::Matrix4d::Identity()), this->K_eigen, this->mDistCoef, keypoints, undistorted_kps, descriptors, depth, 0, frame, this->voc);
            break;
        case 1:
            this->prev_kf = this->current_kf;
            this->current_kf = new KeyFrame(this->prev_kf, keypoints, undistorted_kps, descriptors, depth);
            break;
        case 2:
            this->prev_prev_kf = this->prev_kf;
            this->prev_kf = this->current_kf;
            this->current_kf = new KeyFrame(this->prev_kf, keypoints, undistorted_kps, descriptors, depth);
            break;
        default: 
            if (!this->prev_prev_kf->isKeyFrame)
                delete this->prev_prev_kf;
            this->prev_prev_kf = this->prev_kf;
            this->prev_kf = this->current_kf;
            this->current_kf = new KeyFrame(this->prev_kf, keypoints, undistorted_kps, descriptors, depth);
            break;
    }
    this->velocity = GetVelocityNextFrame();
    frames_seen++;
}

void Tracker::TrackingWasLost()
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
    this->current_kf->set_keyframe_position(this->velocity * this->prev_kf->Tcw);
    auto start = high_resolution_clock::now();
    matcher->match_frame_reference_frame(this->current_kf, this->reference_kf);
    auto end = high_resolution_clock::now();
    this->orb_matching_time += duration_cast<milliseconds>(end - start).count();

    if ((int)this->current_kf->mp_correlations.size() < NR_MAP_POINTS_TRACKED_BETWEEN_FRAMES)
    {
        std::cout << this->current_kf->mp_correlations.size() << " REFERENCE FRAME N A URMARIT SUFICIENTE MAP POINTS PENTRU OPTIMIZARE\n";
        this->TrackingWasLost();
    }
    start = high_resolution_clock::now();
    this->motionOnlyBA->solve_ceres(this->current_kf, this->prev_kf, false);
    end = high_resolution_clock::now();
    this->motion_only_ba_time += duration_cast<milliseconds>(end - start).count();
    if ((int)this->current_kf->mp_correlations.size() < NR_MAP_POINTS_TRACKED_BETWEEN_FRAMES)
    {
        std::cout << this->current_kf->mp_correlations.size() << " REFERENCE FRAME NU A URMARIT SUFICIENTE INLIERE MAP POINTS\n";
        this->TrackingWasLost();
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
    if ((int)this->current_kf->mp_correlations.size() < NR_MAP_POINTS_TRACKED_BETWEEN_FRAMES)
    {
        matcher->match_consecutive_frames(this->current_kf, this->prev_kf, 2 * window);
        if ((int)this->current_kf->mp_correlations.size() < NR_MAP_POINTS_TRACKED_BETWEEN_FRAMES)
        {
            std::cout << this->current_kf->mp_correlations.size() << "  URMARIREA INTRE FRAME-URI INAINTE DE OPTIMIZARE NU A FUNCTIONAT\n";
            this->TrackingWasLost();
        }
    }
    start = high_resolution_clock::now();
    this->motionOnlyBA->solve_ceres(this->current_kf, this->prev_kf, false);
    end = high_resolution_clock::now();
    this->motion_only_ba_time += duration_cast<milliseconds>(end - start).count();
    if ((int)this->current_kf->mp_correlations.size() < NR_MAP_POINTS_TRACKED_BETWEEN_FRAMES)
    {
        std::cout << " \nURMARIREA INTRE FRAME-URI NU A FUNCTIONAT DUPA OPTIMIZARE\n";
        this->TrackingWasLost();
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
    if ((int)this->current_kf->mp_correlations.size() < NR_MAP_POINTS_TRACKED_MAP_LOW)
    {
        std::cout << " \nPRREA PUTINE PUNCTE PROIECTATE DE LOCAL MAP INAINTE DE OPTIMIZARE\n";
        return;
    }
    start = high_resolution_clock::now();
    this->motionOnlyBA->solve_ceres(this->current_kf, this->prev_kf, false);
    end = high_resolution_clock::now();
    this->motion_only_ba_time += duration_cast<milliseconds>(end - start).count();
    int minim_number_points_necessary = mapp->keyframes.size() <= 2 ? NR_MAP_POINTS_TRACKED_MAP_LOW : NR_MAP_POINTS_TRACKED_MAP_HIGH;
    if ((int)this->current_kf->mp_correlations.size() < minim_number_points_necessary)
    {
        std::cout << this->current_kf->mp_correlations.size() << " PREA PUTINE PUNCTE PROIECTATE CARE NU SUNT OUTLIERE DE CATRE LOCAL MAP\n";
        return;
    }
}

KeyFrame *Tracker::tracking(Mat frame, Mat depth)
{
    auto start = high_resolution_clock::now();
    this->GetNextFrame(frame, depth);
    if (frames_seen == 1)
    {
        this->current_kf->isKeyFrame = true;
        this->reference_kf = this->current_kf;
        std::cout << "PRIMUL KEYFRAME VA FI ADAUGAT\n";
        return this->current_kf;
    }

    if (this->current_kf->current_idx - this->reference_kf->current_idx <= 2)
    {
        this->TrackReferenceKeyFrame();
        std::cout << "URMARIT CU AJUTORUL TRACK REFERENCE KEY FRAME\n";
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
    this->current_kf->isKeyFrame = this->Is_KeyFrame_needed();

    if (this->current_kf->isKeyFrame)
    {
        this->reference_kf = this->current_kf;
    }

    this->current_kf->reference_kf = this->reference_kf;
    return this->current_kf;
}