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

Sophus::SE3d Tracker::TrackWithLastFrame(std::vector<std::pair<MapPoint *, Feature*>>& matches, vector<int> &inliers)
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
    Eigen::Matrix3d rotationMatrix = this->current_kf->Tiw.rotationMatrix();
    cv::Mat rotationMat(3, 3, CV_64F);
    cv::eigen2cv(rotationMatrix, rotationMat);
    cv::Rodrigues(rotationMat, r);
    Eigen::Vector3d translationVector = this->current_kf->Tiw.translation();
    t = (cv::Mat_<double>(3, 1) << translationVector(0), translationVector(1), translationVector(2));
    // pag 160 - slambook.en
    cv::solvePnPRansac(points_in3d, points_in2d, K, Mat(), r, t, true, this->ransac_iteration, this->optimizer_window,
    this->ransac_confidence, inliers);
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

void Tracker::reject_outlier(std::unordered_map<MapPoint *, Feature*>& matches, std::unordered_map<MapPoint*, Feature*>& outliers) {
    // std::cout << matches.size() << " nr map point matched inainte de rejectare outliere\n";
    for (auto it = outliers.begin(); it != outliers.end(); it++) {
        if (matches.find(it->first) == matches.end()) continue;
        matches.erase(it->first);
    }
    // std::cout << matches.size() << " nr map point matched dupa rejectare outliere\n";
}

void Tracker::Optimize_Pose_Coordinates(Map mapp, std::vector<std::pair<MapPoint *, Feature*>>& matches, vector<int>& inliers)
{
    // merge these 2, we have to reject the outliers
    std::unordered_map<MapPoint *, Feature*> outliers = get_outliers(matches, inliers);
    std::unordered_map<MapPoint *, Feature*> observed_map_points = mapp.track_local_map(this->current_kf, this->optimizer_window, this->reference_kf);
    reject_outlier(observed_map_points, outliers);
    
    this->frames_tracked += 1;
    if (observed_map_points.size() < this->minim_points_found)
    {
        std::cout << observed_map_points.size() << " atatea map points in momentul in care a crapat\n";
        std::cout << "NOT ENOUGH MAP_POINTS FOUND\n\n";
        std::cout << this->frames_tracked << "\n";
        exit(1);
        // return;
    }
    this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, observed_map_points);
    std::unordered_map<MapPoint *, Feature*> improved_points = mapp.track_local_map(this->current_kf, this->optimizer_window, this->reference_kf);
    this->current_kf->correlate_map_points_to_features_current_frame(improved_points);
}

void Tracker::tracking_was_lost()
{
    std::cout << "TRACKING WAS LOSTT<< \n\n\n";
    std::cout << this->frames_tracked << "\n";
    exit(1);
}

bool Tracker::Is_KeyFrame_needed(std::vector<std::pair<MapPoint *, Feature*>>& matches)
{
    
    this->last_keyframe_added += 1;
    this->keyframes_from_last_global_relocalization += 1;
    bool still_enough_map_points_tracked = matches.size() > this->minim_points_found;  
    bool too_few_map_points_compared_to_kf = reference_kf->nr_map_points / 10 > matches.size(); 
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
    std::cout << this->frames_tracked << " atatea frame-uri urmarite\n";
    this->prev_kf = this->current_kf;
    this->get_current_key_frame(frame, depth);
    // VelocityEstimation();
    std::vector<std::pair<MapPoint *, Feature*>> matches = matcher->match_two_consecutive_frames(this->prev_kf, this->current_kf);
    if (this->Is_KeyFrame_needed(matches))
    {
        std::cout << "DAAA UN KEYFRAME A FOST ADAUGAT\n";
        mapp.add_new_keyframe(this->current_kf);
        this->reference_kf = this->current_kf;
        this->last_keyframe_added = 0;
        this->keyframes_from_last_global_relocalization = 0;
        // exit(1);
        std::cout << "\n\n";
        return;
    } else if(matches.size() < this->minim_points_found)
    {
        std::cout << matches.size() << " puncte au fost matchuite\n\n";
        this->tracking_was_lost();
        return;
    }
    std::vector<int> inliers;
    // print_pose(this->current_kf->Tiw, "inainte de optimizare");

    this->current_kf->Tiw = TrackWithLastFrame(matches, inliers);

    // std::cout << inliers.size() << " inliers found in algorithm\n";
    // print_pose(this->current_kf->Tiw, "dupa estimarea initiala");

    Optimize_Pose_Coordinates(mapp,  matches, inliers);
    // print_pose(this->current_kf->Tiw, "dupa optimizarea BA");
}