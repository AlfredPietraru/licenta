#include "../include/Tracker.h"

void print_pose(Sophus::SE3d pose, std::string message) {
    for (int i = 0; i < 7; i++) {
        std::cout << pose.data()[i] << " ";
    }
    std::cout << message << "\n";
}


void Tracker::get_current_key_frame(Mat frame, Mat depth)
{
    std::vector<KeyPoint> keypoints = this->fmf->extract_keypoints(frame);

    cv::Mat descriptors = this->fmf->compute_descriptors(frame, keypoints);
    // std::cout << keypoints.size() << " " << descriptors.size() << " keypoints and descriptors \n";
    Mat img2;
    Sophus::SE3d pose_estimation = (this->prev_kf == nullptr) ? this->initial_pose : this->prev_kf->Tiw;
    int idx = (this->prev_kf == nullptr) ? 0 : this->reference_kf->idx + 1;
    this->current_kf = new KeyFrame(pose_estimation, this->K_eigen, keypoints, descriptors, depth, idx, frame);
}


Map Tracker::initialize(Mat frame, Mat depth, Config cfg)
{
    this->get_current_key_frame(frame, depth);
    Map mapp = Map(this->matcher, this->current_kf, cfg);
    this->reference_kf = this->current_kf;
    return mapp;
}

Sophus::SE3d Tracker::TrackWithLastFrame(std::unordered_map<MapPoint *, Feature*>& matches)
{
    vector<Point3d> points_in3d;
    vector<Point2d> points_in2d;
    for (std::pair<MapPoint *, Feature*> pair : matches)
    {
        MapPoint *mp = pair.first;
        cv::KeyPoint kp = pair.second->get_key_point();
        points_in3d.push_back(Point3d(mp->wcoord(0), mp->wcoord(1), mp->wcoord(2)));
        points_in2d.push_back(Point2d( kp.pt.x, kp.pt.y));
    }
    // std::cout << points_in3d.size() << " " << points_in2d.size() << " puncte gasite pentru alogirtmul pnp\n";
    cv::Mat r, t, R;
    Eigen::Matrix3d rotationMatrix = this->prev_kf->Tiw.rotationMatrix();
    cv::Mat rotationMat(3, 3, CV_64F);
    cv::eigen2cv(rotationMatrix, rotationMat);
    cv::Rodrigues(rotationMat, r);
    Eigen::Vector3d translationVector = this->prev_kf->Tiw.translation();
    t = (cv::Mat_<double>(3, 1) << translationVector(0), translationVector(1), translationVector(2));
    // pag 160 - slambook.en
    std::vector<int> inliers;
    cv::solvePnPRansac(points_in3d, points_in2d, K, Mat(), r, t, true, this->ransac_iteration, this->optimizer_window,
    this->ransac_confidence, inliers);
    // cv::solvePnP(points_in3d, points_in2d, K, Mat(), r, t, true, cv::SOLVEPNP_EPNP);
    // std::cout << inliers.size() << " inliere intalnite\n";
    cv::Rodrigues(r, R);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Eigen::Vector3d t_eigen;
    t_eigen << t.at<double>(0), t.at<double>(1), t.at<double>(2);
    return Sophus::SE3d(R_eigen, t_eigen);
}

std::unordered_map<MapPoint *, Feature*> Tracker::get_outliers(std::vector<std::pair<MapPoint *, Feature*>>& matches,
         vector<int>& inliers) {
    std::unordered_map<MapPoint *, Feature*> res;
    int inlier_idx = 0;
    for (int i = 0; i < matches.size(); i++) {
        if (i == inliers[inlier_idx]) {
            inlier_idx++;
            continue;
        }
        res.insert(matches[i]); 
    }
    return res;
} 

void Tracker::tracking_was_lost()
{
    std::cout << "TRACKING WAS LOSTT<< \n\n\n";
    std::cout << this->frames_tracked << "\n";
    exit(1);
}

bool Tracker::Is_KeyFrame_needed(std::unordered_map<MapPoint *, Feature*>& matches)
{
    this->last_keyframe_added += 1;
    this->keyframes_from_last_global_relocalization += 1;
    bool still_enough_map_points_tracked = matches.size() > this->minim_points_found;  
    bool too_few_map_points_compared_to_kf = reference_kf->map_points.size() / 10 > matches.size(); 
    bool no_global_relocalization = this->keyframes_from_last_global_relocalization > 20;
    bool no_recent_keyframe_added = this->last_keyframe_added > 20;
    return no_global_relocalization && no_recent_keyframe_added && still_enough_map_points_tracked && too_few_map_points_compared_to_kf;
}

void Tracker::VelocityEstimation()
{
    Sophus::SE3d mVelocity = this->current_kf->Tiw * this->prev_kf->Tiw.inverse();
    this->current_kf->Tiw = mVelocity * this->prev_kf->Tiw;
}

void Tracker::tracking(Mat frame, Mat depth, Map& mapp)
{
    // std::cout << this->frames_tracked << " atatea frame-uri urmarite\n";
    this->prev_kf = this->current_kf;
    this->get_current_key_frame(frame, depth);
    this->frames_tracked += 1;
    // VelocityEstimation();
    std::unordered_map<MapPoint *, Feature*> matches = matcher->match_frame_map_points(this->current_kf, this->prev_kf->map_points);
    if (matches.size() < 20)
    {
        std::cout << matches.size() << " atatea map points in momentul in care a crapat\n";
        this->tracking_was_lost();
        std::cout << "NOT ENOUGH MAP_POINTS FOUND\n\n";
        std::cout << this->frames_tracked << "\n";
        exit(1);
    }
    this->current_kf->Tiw = TrackWithLastFrame(matches);
    mapp.track_local_map(this->current_kf, matches, this->optimizer_window);
    this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, matches, this->minim_points_found);
    this->current_kf->correlate_map_points_to_features_current_frame(matches);
    //  trebuie de verificat aici\n;
    // std::cout << matches.size() << " gasite dupa proiectare local Map\n";
    if (this->Is_KeyFrame_needed(matches))
    {
        std::cout << "DAAA UN KEYFRAME A FOST ADAUGAT\n\n\n";
        mapp.add_new_keyframe(this->current_kf);
        this->reference_kf = this->current_kf;
        this->last_keyframe_added = 0;
        this->keyframes_from_last_global_relocalization = 0;
        return;
    } else if(matches.size() < this->minim_points_found)
    {
        std::cout << matches.size() << " puncte au fost matchuite\n\n";
        this->tracking_was_lost();
        return;
    }
}