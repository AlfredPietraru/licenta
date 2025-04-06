#include "../include/Tracker.h"


void Tracker::VelocityEstimation()
{
    Sophus::SE3d mVelocity = this->current_kf->Tiw * this->prev_kf->Tiw.inverse();
    this->current_kf->Tiw = mVelocity * this->prev_kf->Tiw;
}

void compute_difference_between_positions(const Sophus::SE3d &estimated, const Sophus::SE3d &ground_truth)
{
    Sophus::SE3d relative = ground_truth.inverse() * estimated;
    double APE = relative.log().norm();

    Eigen::Vector3d angle_axis = relative.so3().log(); // Rotation vector (Lie Algebra)
    double angle_diff_x = std::abs(angle_axis.x()) * 180.0 / M_PI;
    double angle_diff_y = std::abs(angle_axis.y()) * 180.0 / M_PI;
    double angle_diff_z = std::abs(angle_axis.z()) * 180.0 / M_PI;

    Eigen::Quaterniond q_rel = relative.unit_quaternion();
    double total_angle_diff = 2.0 * std::acos(q_rel.w()) * 180.0 / M_PI;
    if (total_angle_diff > 180.0) {
        total_angle_diff = 360.0 - total_angle_diff;
    }

    Eigen::Vector3d t_rel = relative.translation();
    double translation_diff = t_rel.norm();

    std::cout << "APE score: " << APE << "\n";
    std::cout << "Rotation Difference (Total): " << total_angle_diff << " degrees\n";
    std::cout << "Rotation Difference (X): " << angle_diff_x << " degrees\n";
    std::cout << "Rotation Difference (Y): " << angle_diff_y << " degrees\n";
    std::cout << "Rotation Difference (Z): " << angle_diff_z << " degrees\n";
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
    std::vector<KeyPoint> keypoints = this->fmf->extract_keypoints(frame);
    cv::Mat descriptors = this->fmf->compute_descriptors(frame, keypoints);
    cv::Mat img2;
    std::cout << keypoints.size() << " " << descriptors.size() << "\n";
    cv::drawKeypoints(frame, keypoints, img2, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Display window", img2);
    cv::waitKey(100);

    Sophus::SE3d pose_estimation = (this->prev_kf == nullptr) ? this->initial_pose : this->prev_kf->Tiw;
    int idx = (this->prev_kf == nullptr) ? 0 : this->reference_kf->idx + 1;
    this->current_kf = new KeyFrame(pose_estimation, this->K_eigen, keypoints, descriptors, depth, idx, frame, this->voc);
    this->frames_tracked += 1;
}

void Tracker::initialize(Mat frame, Mat depth, Map& mapp)
{
    this->get_current_key_frame(frame, depth);
    this->reference_kf = this->current_kf;
    mapp.add_new_keyframe(this->reference_kf);
}



void Tracker::tracking_was_lost()
{
    std::cout << "TRACKING WAS LOSTT<< \n\n\n";
    std::cout << this->frames_tracked << "\n";
    exit(1);
}

bool Tracker::Is_KeyFrame_needed(std::unordered_map<MapPoint *, Feature *> &matches)
{
    this->last_keyframe_added += 1;
    this->keyframes_from_last_global_relocalization += 1;
    bool still_enough_map_points_tracked = matches.size() > this->minim_points_found;
    bool too_few_map_points_compared_to_kf = reference_kf->map_points.size() / 10 > matches.size();
    bool no_global_relocalization = this->keyframes_from_last_global_relocalization > 20;
    bool no_recent_keyframe_added = this->last_keyframe_added > 20;
    return no_global_relocalization && no_recent_keyframe_added && still_enough_map_points_tracked && too_few_map_points_compared_to_kf;
}


void Tracker::tracking(Mat frame, Mat depth, Map &mapp, Sophus::SE3d ground_truth_pose) {
    this->prev_kf = this->current_kf;
    this->get_current_key_frame(frame, depth);
    print_pose(ground_truth_pose, "ground truth pose\n");
    std::unordered_map<MapPoint *, Feature *> matches = matcher->match_frame_reference_frame(this->current_kf, this->reference_kf, this->voc);
    // std::unordered_map<MapPoint *, Feature *> matches = matcher->match_frame_map_points(this->current_kf, this->prev_kf);
    std::cout << matches.size() << " atatea map points initial obtinute\n";
    
    vector<cv::KeyPoint> keypoints;
    for (auto it = matches.begin(); it != matches.end(); it++) {
        keypoints.push_back(it->second->get_key_point());
    }
    cv::Mat img2, img3;
    cv::drawKeypoints(this->current_kf->frame, this->current_kf->get_all_keypoints(), img2, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
    cv::drawKeypoints(img2, keypoints, img3, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT); 
    cv::imshow("Display window", img3);
    cv::waitKey(0);

    if (matches.size() < 20) {
        std::cout << matches.size() << " atatea map points in momentul in care a crapat\n";
        this->tracking_was_lost();
        std::cout << "NOT ENOUGH MAP_POINTS FOUND\n\n";
        std::cout << this->frames_tracked << "\n";
        exit(1);
    }

    this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, matches);
    
    print_pose(this->current_kf->Tiw, " estimare pozitie dupa prima aproximare\n");
    std::unordered_map<MapPoint *, Feature *> new_matches = mapp.track_local_map(this->current_kf, matches);
    if (new_matches.size() < 50) {
        std::cout << new_matches.size() << " prea putine puncte proiectate de catre local map ceva nu e bine\n";
        exit(1);
    }
    std::cout << new_matches.size() << " puncte reproiectate de catre local map\n";
    this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, new_matches);
    this->current_kf->correlate_map_points_to_features_current_frame(new_matches);
    compute_difference_between_positions(this->current_kf->Tiw, ground_truth_pose);
    
    if (this->Is_KeyFrame_needed(matches))
    {
        std::cout << "DAAA UN KEYFRAME A FOST ADAUGAT\n\n\n";
        mapp.add_new_keyframe(this->current_kf);
        this->reference_kf = this->current_kf;
        this->last_keyframe_added = 0;
        this->keyframes_from_last_global_relocalization = 0;
        return;
    }
    else if (matches.size() < this->minim_points_found)
    {
        std::cout << matches.size() << " puncte au fost matchuite\n\n";
        this->tracking_was_lost();
        return;
    }
}