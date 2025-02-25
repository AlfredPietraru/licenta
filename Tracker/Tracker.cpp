#include "Tracker.h"

void Tracker::set_prev_frame() {
    this->prev_frame = this->current_frame;
    this->prev_depth = this->current_depth;
    this->prev_des = this->current_des;
    this->prev_kps = this->current_kps;
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

std::pair<Graph, vector<MapPoint>> Tracker::initialize() {
    vector<MapPoint> map_points;
    get_next_image();
    compute_features_descriptors();
    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity(); 
    Eigen::Vector3d camera_center = compute_camera_center(identity);
    for (int i = 0; i < this->current_kps.size(); i++) {
        float depth = this->current_depth.at<float>((int)this->current_kps[i].pt.x, (int)this->current_kps[i].pt.y);
        if (depth < 0.001) continue;
        map_points.push_back(MapPoint(this->current_kps[i], depth, identity, camera_center, this->current_des.row(i)));
    }
    this->last_keyframe = KeyFrame(identity, K, this->current_kps, this->current_des, this->current_depth);
    Graph graph = Graph(this->last_keyframe);
    return std::pair<Graph, vector<MapPoint>>(graph, map_points);
}

std::pair<vector<MapPoint>, vector<KeyPoint>> Tracker::correlation_map_point_keypoint_idx(Eigen::Matrix4d& pose, vector<MapPoint> map_points) {
        vector<MapPoint> observed_map_points;
        vector<KeyPoint> kps;
        for (MapPoint mp : map_points) {
            std::pair<float, float> camera_coordinates = fromWorldToCamera(pose, this->current_depth, mp.wcoord);
            if (camera_coordinates.first < 0  || camera_coordinates.second < 0) continue;
            float u = camera_coordinates.first;
            float v = camera_coordinates.second;
            int min_hamming_distance = 10000;
            int current_hamming_distance = -1;
            KeyPoint right_kp = KeyPoint();
            for (int i = 0; i < this->current_kps.size(); i++) {
                if (this->current_kps[i].pt.x - WINDOW > u || this->current_kps[i].pt.x + WINDOW < u) continue;
                if (this->current_kps[i].pt.y - WINDOW > v || this->current_kps[i].pt.y + WINDOW < v) continue;
                current_hamming_distance = ComputeHammingDistance(mp.orb_descriptor, this->current_des.row(i));
                if (current_hamming_distance < min_hamming_distance) {
                    min_hamming_distance = current_hamming_distance;
                    right_kp = this->current_kps[i];
                } 
            }
            if (current_hamming_distance == -1) continue;
            observed_map_points.push_back(mp);
            kps.push_back(right_kp);
        }
        return std::pair<vector<MapPoint>, vector<KeyPoint>>(observed_map_points, kps);
    }

Eigen::Matrix4d Tracker::TrackWithLastFrame(std::vector<MapPoint> map_points) {
    vector<DMatch> good_matches = this->match_features_last_frame();
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
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            T(i,j) = R.at<double>(i, j);
        }
    }
    for (int i = 0; i < 3; i++) {
        T(i, 3) = t.at<double>(i);
    }
    std::pair<vector<MapPoint>, vector<KeyPoint>> pairr = correlation_map_point_keypoint_idx(T, map_points);
    this->bundleAdjustment = BundleAdjustment(pairr.first, pairr.second, T);
    this->bundleAdjustment.solve();
    cout << T << "\n";
    return this->bundleAdjustment.return_optimized_pose();
     
}


void Tracker::tracking(std::vector<MapPoint> map_points) {
    set_prev_frame();
    get_next_image();
    compute_features_descriptors();
    Eigen::Matrix4d T = TrackWithLastFrame(map_points);
    cout << T << "\n";

}