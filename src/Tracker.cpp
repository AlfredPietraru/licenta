#include "../include/Tracker.h"

void Tracker::get_current_key_frame() {
    Mat frame;
    cap.read(frame);
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(224, 224));
    Mat blob = cv::dnn::blobFromImage(resized, 1, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat depth = net.forward().reshape(1, 224);
    depth = depth * 10000;
    std::vector<KeyPoint> keypoints = this->fmf->extract_keypoints(resized);
    Mat img2;
    cv::drawKeypoints(resized, keypoints, img2, Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
    imshow("Display window", img2);
    waitKey(0);
    cv::Mat descriptors = this->fmf->compute_descriptors(resized, keypoints);
    std::cout << keypoints.size() << " " << descriptors.size() << " \n"; 
    // exit(1);

    if (this->prev_kf == nullptr) {
        this->current_kf = new KeyFrame(Sophus::SE3d(Eigen::Matrix4d::Identity()), K, keypoints, descriptors, depth, 0); 
    } else {
        this->current_kf = new KeyFrame(this->prev_kf->Tiw, K, keypoints, descriptors, depth, this->reference_kf->idx + 1);
    }
}

Map Tracker::initialize() {
    this->get_current_key_frame();
    Map mapp = Map(this->current_kf);
    this->reference_kf = this->current_kf;
    return mapp;
}

Sophus::SE3d Tracker::TrackWithLastFrame(vector<DMatch> good_matches) {
    vector<Point3d> points_in3d;
    vector<Point2d> points_in2d;
    vector<cv::KeyPoint> kps = this->prev_kf->keypoints;
    cv::Mat depth_matrix = this->prev_kf->depth_matrix;
    vector<cv::KeyPoint> current_kps = this->current_kf->keypoints;
    for (DMatch m : good_matches) {
      if (m.queryIdx >= kps.size() || m.trainIdx >= kps.size()) continue;
      float d = this->prev_kf->depth_matrix.ptr<float>(int(kps[m.queryIdx].pt.y))[int(kps[m.queryIdx].pt.x)];
      if (d <= 0) continue;
      float new_x = (kps[m.queryIdx].pt.x - this->prev_kf->intrisics(0, 2)) * d / this->prev_kf->intrisics(0, 0);
      float new_y = (kps[m.queryIdx].pt.y - this->prev_kf->intrisics(1, 2)) * d / this->prev_kf->intrisics(1, 1);
      points_in3d.push_back(Point3d(new_x, new_y, d));
      points_in2d.push_back(current_kps[m.trainIdx].pt);
    }
    Mat r, t;
    // pag 160 - slambook.en
    cv::solvePnPRansac(points_in3d, points_in2d, convert_from_eigen_to_cv2(K), Mat() , r, t);
    Mat R;
    cv::Rodrigues(r, R);
    Eigen::Matrix3d R_eigen;
    cv::cv2eigen(R, R_eigen);
    Eigen::Vector3d t_eigen;
    t_eigen << t.at<double>(0), t.at<double>(1), t.at<double>(2);
    return Sophus::SE3d(R_eigen,  t_eigen);
}

void Tracker::Optimize_Pose_Coordinates(Map mapp) {
        std::pair<std::vector<MapPoint*>, std::vector<cv::KeyPoint>> observed_map_points = mapp.track_local_map(this->current_kf);
        std::cout << observed_map_points.first.size() << "\n";
        this->frames_tracked += 1;
        if (observed_map_points.first.size() < 15) {
            std::cout << "NOT ENOUGH MAP_POINTS FOUND\n\n";
            std::cout << this->frames_tracked << "\n";
            exit(1);
            // return;

        } 
        this->current_kf->Tiw = this->bundleAdjustment->solve(this->current_kf->Tiw, observed_map_points.first, observed_map_points.second);
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


void Tracker::tracking(Map mapp, vector<KeyFrame*> &key_frames_buffer) {
    this->prev_kf = this->current_kf;
    this->get_current_key_frame();
    vector<DMatch> good_matches = this->fmf->match_features_last_frame(this->current_kf, this->prev_kf);
    // std::cout << good_matches.size() << " good matches found\n";
    if (good_matches.size() < 50) {
        this->tracking_was_lost();
    } else {
        Sophus::SE3d relative_pose_last_2_frames = TrackWithLastFrame(good_matches);
        this->current_kf->Tiw = this->current_kf->Tiw * relative_pose_last_2_frames;
        std::cout << this->current_kf->Tiw.matrix() << "\n"; 
        Optimize_Pose_Coordinates(mapp);
        std::cout << this->current_kf->Tiw.matrix() << "\n";

    }
    if (this->Is_KeyFrame_needed(mapp)) {
        std::cout << "daa adauga un keyframe aicii\n";
        key_frames_buffer.push_back(this->current_kf);
    }
}