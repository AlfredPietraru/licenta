#include "../include/Tracker.h"



void Tracker::get_current_key_frame() {
    Mat frame;
    cap.read(frame);
    Mat blob = cv::dnn::blobFromImage(frame, 1, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat depth = net.forward().reshape(1, 224);
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    fast->detect(frame, keypoints);
    orb->compute(frame, keypoints, descriptors);
    this->current_kf = new KeyFrame(Eigen::Matrix4d::Identity(), K, keypoints, descriptors, depth); 
}

void Tracker::set_prev_key_frame() {
    this->prev_kf = this->current_kf;
}

vector<DMatch> Tracker::match_features_last_frame() {
    vector<DMatch> matches;
    this->matcher->match(this->current_kf->orb_descriptors, this->prev_kf->orb_descriptors, matches);
    vector<DMatch> good_matches; 
    for (auto match : matches) {
        if(match.distance < LIMIT_MATCHING) {
            good_matches.push_back(match);
        }
    }
    return good_matches;
} 

std::pair<Graph, Map> Tracker::initialize() {
    Map mapp = Map();
    vector<MapPoint*> map_points;
    this->get_current_key_frame();
    mapp.add_map_points(this->current_kf);
    Graph graph = Graph(this->current_kf);
    this->last_kf_saved = this->current_kf;
    return std::pair<Graph, Map>(graph, mapp);
}

// e foarte posibil sa nu fie corect de parsat acolo this->last_keyframe, dar nu stiu exact cum altfel ar trebui de parsat
// probabil frame-ul curent cu keypoints si descriptori?
void Tracker::Optimize_Pose_Coordinates(Map mapp) {
        vector<MapPoint*> observed_map_points = mapp.get_map_points(this->current_kf);
        if (observed_map_points.size() < 15) {
            std::cout << "NOT ENOUGH MAP_POINTS FOUND\n\n\n";
            return;
        }
        this->bundleAdjustment = BundleAdjustment(observed_map_points, this->current_kf);
        this->bundleAdjustment.solve();
        this->current_kf->Tiw = this->bundleAdjustment.return_optimized_pose();
    }

void Tracker::TrackWithLastFrame(vector<DMatch> good_matches) {
    vector<Point3d> points_in3d;
    vector<Point2d> points_in2d;
    vector<cv::KeyPoint> kps = this->prev_kf->keypoints;
    cv::Mat depth_matrix = this->prev_kf->depth_matrix;
    for (DMatch m : good_matches) {
      float d = depth_matrix.at<float>((int)kps[m.queryIdx].pt.y, kps[m.queryIdx].pt.x);
      std::cout << d << " ";
      if (d <= 0.01) continue;
      float new_x = (this->prev_kf->keypoints[m.queryIdx].pt.x - X_CAMERA_OFFSET) * d / FOCAL_LENGTH;
      float new_y = (this->prev_kf->keypoints[m.queryIdx].pt.y - Y_CAMERA_OFFSET) * d / FOCAL_LENGTH;
      points_in3d.push_back(Point3d(new_x, new_y, d));
      points_in2d.push_back(this->current_kf->keypoints[m.trainIdx].pt);
    }
    cout << "\n";
    Mat r, t;
    // pag 160 - slambook.en
    cv::solvePnP(points_in3d, points_in2d, convert_from_eigen_to_cv2(K), Mat() , r, t);
    Mat R;
    cv::Rodrigues(r, R);
    this->current_kf->Tiw = compute_pose_matrix(R, t);
}

void Tracker::tracking_was_lost() {
    std::cout << "TRACKING WAS LOSTT<< \n\n\n";
}


bool Tracker::Is_KeyFrame_needed(Map mapp) {
    this->last_keyframe_added  += 1;
    this->keyframes_from_last_global_relocalization += 1;
    bool no_global_relocalization = this->keyframes_from_last_global_relocalization > 20;
    bool no_recent_keyframe_added = this->last_keyframe_added > 20;
    int map_points_seen_in_key_frame = mapp.get_map_points(this->current_kf).size();
    bool current_frame_less_than_80_percent = map_points_seen_in_key_frame > 8 * this->bundleAdjustment.map_points.size();
    return no_global_relocalization && no_recent_keyframe_added && current_frame_less_than_80_percent;
}


void Tracker::tracking(Map mapp, vector<KeyFrame*> &key_frames_buffer) {
    this->set_prev_key_frame();
    this->get_current_key_frame();
    vector<DMatch> good_matches = this->match_features_last_frame();
    if (good_matches.size() < 50) {
        this->tracking_was_lost();
    } else {
        TrackWithLastFrame(good_matches);
        cout << this->current_kf->Tiw << "\n\n";
        Optimize_Pose_Coordinates(mapp);
        cout << this->current_kf->Tiw << "\n\n";
    }
    if (this->Is_KeyFrame_needed(mapp)) {
        key_frames_buffer.push_back(this->current_kf);
    }
}