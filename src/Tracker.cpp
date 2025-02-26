#include "../include/Tracker.h"

void Tracker::set_prev_frame() {
    this->prev_frame = this->current_frame;
    this->prev_depth = this->current_depth;
    this->prev_des = this->current_des;
    this->prev_kps = this->current_kps;
    this->prev_T = this->current_T;
}

void Tracker::get_next_image() {
    Mat frame;
    cap.read(frame);
    Mat blob = cv::dnn::blobFromImage(frame, 1, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat depth = net.forward().reshape(1, 224);
    this->current_frame = frame;
    this->current_depth = depth;

}

vector<DMatch> Tracker::match_features_last_frame() {
    vector<DMatch> matches;
    this->matcher->match(this->prev_des, this->current_des, matches);
    vector<DMatch> good_matches; 
    for (auto match : matches) {
        if(match.distance < LIMIT_MATCHING) {
            good_matches.push_back(match);
        }
    }
    return good_matches;
} 


void Tracker::compute_features_descriptors() {
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    fast->detect(this->current_frame, keypoints);
    orb->compute(this->current_frame, keypoints, descriptors);
    this->current_kps = keypoints;
    this->current_des = descriptors;
} 

std::pair<Graph, Map> Tracker::initialize() {
    Map mapp = Map();
    vector<MapPoint*> map_points;
    get_next_image();
    compute_features_descriptors();
    this->current_T = Eigen::Matrix4d::Identity(); 
    Eigen::Vector3d camera_center = compute_camera_center(this->current_T);
    this->last_keyframe = new KeyFrame(this->current_T, K, this->current_kps, this->current_des, this->current_depth);
    for (int i = 0; i < this->current_kps.size(); i++) {
        float depth = this->current_depth.at<float>((int)this->current_kps[i].pt.x, (int)this->current_kps[i].pt.y);
        if (depth < 0.001) continue;
        map_points.push_back(new MapPoint(this->last_keyframe, this->current_kps[i], depth,
         this->current_T, camera_center, this->current_des.row(i)));
    }
    Graph graph = Graph(this->last_keyframe);
    mapp.add_multiple_map_points(this->last_keyframe, map_points);
    return std::pair<Graph, Map>(graph, mapp);
}

Eigen::Matrix4d Tracker::Optimize_Pose_Coordinates(Eigen::Matrix4d& pose, Map mapp) {
        vector<MapPoint*> observed_map_points = mapp.return_map_points_seen_in_frame(this->last_keyframe, pose, this->current_depth);
        if (observed_map_points.size() < 15) {
            std::cout << "we cooked";
        }
        this->bundleAdjustment = BundleAdjustment(observed_map_points, this->current_kps, 
        this->current_des, this->current_depth, pose);
        this->bundleAdjustment.solve();
        return this->bundleAdjustment.return_optimized_pose();
    }

Eigen::Matrix4d Tracker::TrackWithLastFrame(vector<DMatch> good_matches) {
    vector<Point3d> points_in3d;
    vector<Point2d> points_in2d;
    for (DMatch m : good_matches) {
      float d = this->prev_depth.ptr<float>(int(this->prev_kps[m.queryIdx].pt.y))[int(this->prev_kps[m.queryIdx].pt.x)];
      if (d <= 0.01) continue;
      float new_x = (this->prev_kps[m.queryIdx].pt.x - X_CAMERA_OFFSET) * d / FOCAL_LENGTH;
      float new_y = (this->prev_kps[m.queryIdx].pt.y - Y_CAMERA_OFFSET) * d / FOCAL_LENGTH;
      points_in3d.push_back(Point3d(new_x, new_y, d));
      points_in2d.push_back(this->current_kps[m.trainIdx].pt);
    }
    Mat r, t;
    // pag 160 - slambook.en
    cv::solvePnP(points_in3d, points_in2d, convert_from_eigen_to_cv2(K), Mat() , r, t);
    Mat R;
    cv::Rodrigues(r, R);
    return compute_pose_matrix(R, t);
}

void Tracker::tracking_was_lost() {}


bool Tracker::Is_KeyFrame_needed(Map mapp) {
    this->last_keyframe_added  += 1;
    this->keyframes_from_last_global_relocalization += 1;
    bool no_global_relocalization = this->keyframes_from_last_global_relocalization > 20;
    bool no_recent_keyframe_added = this->last_keyframe_added > 20;
    int map_points_seen_in_key_frame = mapp.get_map_points(this->last_keyframe).size();
    bool current_frame_less_than_80_percent = map_points_seen_in_key_frame > 8 * this->bundleAdjustment.map_points.size();
    return no_global_relocalization && no_recent_keyframe_added && current_frame_less_than_80_percent;
}


void Tracker::tracking(Map mapp, vector<KeyFrame*> &key_frames_buffer) {
    set_prev_frame();
    get_next_image();
    compute_features_descriptors();
    vector<DMatch> good_matches = this->match_features_last_frame();
    if (good_matches.size() < 50) {
        std::cout << "you cooked_bro\n"; 
        this->tracking_was_lost();
    } else {
        this->current_T = TrackWithLastFrame(good_matches);
        cout << this->current_T << "\n";
        this->current_T = Optimize_Pose_Coordinates(this->current_T, mapp);
        cout << this->current_T << "\n";
    }
    if (this->Is_KeyFrame_needed(mapp)) {
        KeyFrame *new_keyframe = new KeyFrame(this->current_T, this->K,
                 this->current_kps, this->current_des, this->current_depth);
        key_frames_buffer.push_back(new_keyframe);
    }
}