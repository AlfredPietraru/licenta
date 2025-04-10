#include "../include/Tracker.h"


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
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    (*this->extractor)(gray, cv::Mat(), keypoints, descriptors);
    if (this->prev_kf == nullptr && this->current_kf != nullptr) {
        this->prev_kf = this->current_kf;
        this->current_kf = new KeyFrame(this->prev_kf->Tiw, this->K_eigen, keypoints, descriptors, depth, 1, gray, this->voc);
        return;
    }

    Sophus::SE3d pose_estimation = this->current_kf->Tiw * this->prev_kf->Tiw.inverse() * this->current_kf->Tiw;
    this->prev_kf = this->current_kf;
    this->current_kf = new KeyFrame(pose_estimation, this->K_eigen, keypoints, descriptors, depth, 
        this->prev_kf->current_idx + 1, gray, this->voc);
}

void Tracker::initialize(Mat frame, Mat depth, Map& mapp, Sophus::SE3d pose)
{
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
    (*this->extractor)(gray,cv::Mat(),keypoints, descriptors);
    this->current_kf = new KeyFrame(pose, this->K_eigen, keypoints, descriptors, depth, 0, gray, this->voc);

    this->reference_kf = this->current_kf;
    mapp.add_first_keyframe(this->reference_kf);
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
    bool c5 = this->current_kf->check_possible_close_points_generation() > 70;
    return c1 && c2 && c3 && c4 && true;
}


std::unordered_map<MapPoint *, Feature *> Tracker::TrackReferenceKeyFrame() {
    std::unordered_map<MapPoint *, Feature *> matches = matcher->match_frame_reference_frame(this->current_kf, this->reference_kf);
    if (matches.size() < 15) {
        std::cout << matches.size() << "REFERENCE FRAME N A URMARIT SUFICIENTE MAP POINTS PENTRU OPTIMIZARE\n";
        std::cout << this->current_kf->current_idx << "\n";
        exit(1);
    }
    this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, matches);
    int out = 0; 
    for (auto it = matches.begin(); it != matches.end(); it++) {
        if (this->current_kf->check_map_point_outlier(it->first)) continue;
        out += 1;
    }
    if (out < 10) {
        std::cout << out << "REFERENCE FRAME NU A URMARIT SUFICIENTE INLIERE MAP POINTS\n";
        std::cout << this->current_kf->current_idx << "\n";
        exit(1);
    }
    return matches; 
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
    this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, matches);
    int out = 0;
    for (auto it = matches.begin(); it != matches.end(); it++) {
        MapPoint *mp = it->first;
        if (this->current_kf->check_map_point_outlier(mp)) continue;
        out++;
    }

    if (out < 10) {
        std::cout << "\nURMARIREA INTRE FRAME-URI NU A FUNCTIONAT DUPA OPTIMIZARE\n";
        return matches;
    }
    return matches; 
}

// de verificat local map bine de tot
std::unordered_map<MapPoint*, Feature*> Tracker::TrackLocalMap(Map &mapp) {
    std::unordered_map<MapPoint *, Feature *> new_matches = mapp.track_local_map(this->current_kf);
    this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, new_matches);
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

KeyFrame *Tracker::tracking(Mat frame, Mat depth, Map &mapp, Sophus::SE3d ground_truth_pose) {
    this->get_current_key_frame(frame, depth);
    std::unordered_map<MapPoint *, Feature *> matches;
    if (this->current_kf->current_idx - this->reference_kf->current_idx <= 2) {
        std::cout << "inca urmarit cu ajutorul track reference frame\n";
        matches = this->TrackReferenceKeyFrame();
    } else {
        matches = this->TrackConsecutiveFrames();
        // std::cout << matches.size() << "\n";
        if (matches.size() < 20) {
            std::cout << "intre frame-uri nu s-a gasit suficient\n";
            matches = this->TrackReferenceKeyFrame();
        }
    }
    this->current_kf->correlate_map_points_to_features_current_frame(matches);
    std::unordered_map<MapPoint *, Feature *>  new_matches = this->TrackLocalMap(mapp);
    this->current_kf->correlate_map_points_to_features_current_frame(new_matches);
    compute_difference_between_positions(this->current_kf->Tiw, ground_truth_pose);
    this->current_kf->debug_keyframe(100, matches, new_matches);
    return this->current_kf;
}