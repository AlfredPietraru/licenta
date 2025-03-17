#include "../include/Tracker.h"

void Tracker::get_current_key_frame(Mat frame, Mat depth) {
    std::vector<KeyPoint> keypoints = this->fmf->extract_keypoints(frame);
    
    
    cv::Mat descriptors = this->fmf->compute_descriptors(frame, keypoints);
    std::cout << keypoints.size() << " " << descriptors.size() << " keypoints and descriptors \n"; 
    Mat img2;
    Sophus::SE3d pose_estimation = (this->prev_kf == nullptr) ? this->initial_pose : this->prev_kf->Tiw; 
    int idx = (this->prev_kf == nullptr) ? 0 : this->reference_kf->idx + 1;
    this->current_kf = new KeyFrame(pose_estimation, this->K_eigen, keypoints, descriptors, depth, idx, frame);
}

Map Tracker::initialize(Mat frame, Mat depth, Config cfg) { 
    this->get_current_key_frame(frame, depth);
    Map mapp = Map(this->current_kf, cfg);
    this->reference_kf = this->current_kf;
    return mapp;
}

Sophus::SE3d Tracker::TrackWithLastFrame(std::vector<std::pair<MapPoint*, cv::KeyPoint>> matches) {
    vector<Point3d> points_in3d;
    vector<Point2d> points_in2d;
    vector<cv::KeyPoint> prev_kps = this->prev_kf->get_all_keypoints();
    cv::Mat depth_matrix = this->prev_kf->depth_matrix;
    vector<cv::KeyPoint> current_kps = this->current_kf->get_all_keypoints();
    for (std::pair<MapPoint*, cv::KeyPoint> pair : matches) {
        MapPoint *mp = pair.first;
        cv::KeyPoint kp = pair.second;
        points_in3d.push_back(Point3d(mp->wcoord(0), mp->wcoord(1), mp->wcoord(2)));
        points_in2d.push_back(Point2d(kp.pt.x, kp.pt.y));
    }
    std::cout  << points_in3d.size() << " " << points_in2d.size() << " puncte gasite pentru alogirtmul pnp\n";
    Mat r, t;
    // pag 160 - slambook.en
    vector<int> inliers;
    cv::solvePnP(points_in3d, points_in2d, K, cv::Mat(), r, t, false);
    cv::solvePnPRansac(points_in3d, points_in2d, K, Mat() , r, t, true, this->ransac_iteration, this->optimizer_window, this->ransac_confidence, inliers); 
    std::cout << inliers.size() << " inliers found in algorithm\n";
    Mat R;
    cv::Rodrigues(r, R);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Eigen::Vector3d t_eigen;
    t_eigen << t.at<double>(0), t.at<double>(1), t.at<double>(2);
    return Sophus::SE3d(R_eigen,  t_eigen);
}

void Tracker::Optimize_Pose_Coordinates(Map mapp, cv::Mat frame) {
        std::pair<std::vector<MapPoint*>, std::vector<cv::KeyPoint>> observed_map_points = mapp.track_local_map(this->current_kf, this->optimizer_window);
        this->frames_tracked += 1;
        if (observed_map_points.first.size() < 15) {
            std::cout << "NOT ENOUGH MAP_POINTS FOUND\n\n";
            std::cout << this->frames_tracked << "\n";
            exit(1);
            // return;
        } 
        this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf, observed_map_points.first, observed_map_points.second);
}

void Tracker::tracking_was_lost() {
    std::cout << "TRACKING WAS LOSTT<< \n\n\n";
}


bool Tracker::Is_KeyFrame_needed(Map mapp) {
    this->last_keyframe_added  += 1;
    this->keyframes_from_last_global_relocalization += 1;
    bool no_global_relocalization = this->keyframes_from_last_global_relocalization > 20;
    bool no_recent_keyframe_added = this->last_keyframe_added > 20;
    return no_global_relocalization && no_recent_keyframe_added;
}

void Tracker::VelocityEstimation() {
    Sophus::SE3d mVelocity = this->current_kf->Tiw * this->prev_kf->Tiw.inverse();
    this->current_kf->Tiw = mVelocity * this->prev_kf->Tiw;
}


void Tracker::tracking(Mat frame, Mat depth, Map mapp, vector<KeyFrame*> &key_frames_buffer) {
    this->prev_kf = this->current_kf;
    this->get_current_key_frame(frame, depth);
    // VelocityEstimation();
    std::vector<std::pair<MapPoint*, cv::KeyPoint>> matches = matcher->match_two_consecutive_frames(this->prev_kf,
             this->current_kf, this->optimizer_window);
    std::cout << matches.size() << " puncte au fost matchuite\n\n";
    if (matches.size() < 20) {
        this->tracking_was_lost();
    } else {
        for (int i = 0; i < 7; i++) {
            std::cout << this->current_kf->Tiw.data()[i] << " "; 
        }
        std::cout << " inainte de optimizare \n";
        Sophus::SE3d relative_pose_last_2_frames = TrackWithLastFrame(matches);
        // for (int i = 0; i < 7; i++) {
        //     std::cout << relative_pose_last_2_frames.data()[i] << " "; 
        // }
        // std::cout << " modificarea intre frame-uri \n";
        this->current_kf->Tiw = relative_pose_last_2_frames;
        for (int i = 0; i < 7; i++) {
            std::cout << this->current_kf->Tiw.data()[i] << " "; 
        }
        std::cout << " dupa estimarea initiala \n";
        Optimize_Pose_Coordinates(mapp, frame);
        for (int i = 0; i < 7; i++) {
            std::cout << this->current_kf->Tiw.data()[i] << " "; 
        }
        std::cout << " dupa optimizarea BA\n\n";
        // exit(1);
    }
    if (this->Is_KeyFrame_needed(mapp)) {
        std::cout << "daa adauga un keyframe aicii\n";
        key_frames_buffer.push_back(this->current_kf);
    }
}


// Sophus::SE3d Tracker::TrackWithLastFrame(vector<DMatch> good_matches) {
//     vector<Point3d> points_in3d;
//     vector<Point2d> points_in2d;
//     vector<cv::KeyPoint> prev_kps = this->prev_kf->get_all_keypoints();
//     cv::Mat depth_matrix = this->prev_kf->depth_matrix;
//     vector<cv::KeyPoint> current_kps = this->current_kf->get_all_keypoints();
//     for (DMatch m : good_matches) {
//       if (m.queryIdx >= prev_kps.size() || m.trainIdx >= current_kps.size()) continue;
//         float dd = this->prev_kf->compute_depth_in_keypoint(prev_kps[m.queryIdx]);
//         // std::cout << dd << " ";
//       if (dd <= 0) continue;
//       float new_x = (prev_kps[m.queryIdx].pt.x - this->prev_kf->K(0, 2)) * dd / this->prev_kf->K(0, 0);
//       float new_y = (prev_kps[m.queryIdx].pt.y - this->prev_kf->K(1, 2)) * dd / this->prev_kf->K(1, 1);
//       points_in3d.push_back(Point3d(new_x, new_y, dd));
//       points_in2d.push_back(Point2d(current_kps[m.trainIdx].pt.x, current_kps[m.trainIdx].pt.y));
//     }
//     std::cout  << points_in3d.size() << " " << points_in2d.size() << " puncte gasite pentru alogirtmul pnp\n";
//     Mat r, t;
//     // pag 160 - slambook.en
//     vector<int> inliers;
//     cv::solvePnPRansac(points_in3d, points_in2d, K, Mat() , r, t, true, this->ransac_iteration, 
//         this->optimizer_window, this->ransac_confidence, inliers); 
//     std::cout << inliers.size() << " inliers found in algorithm\n";
//     Mat R;
//     cv::Rodrigues(r, R);
//     Eigen::Matrix3d R_eigen;
//     cv::cv2eigen(R, R_eigen);
//     Eigen::Vector3d t_eigen;
//     t_eigen << t.at<double>(0), t.at<double>(1), t.at<double>(2);
//     return Sophus::SE3d(R_eigen,  t_eigen);
// }