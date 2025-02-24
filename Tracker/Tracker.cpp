#include "Tracker.h"

std::pair<Mat, Mat> Tracker::get_next_image() {
    Mat frame;
    cap.read(frame);
    Mat blob = cv::dnn::blobFromImage(frame, 1, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    Mat depth = net.forward().reshape(1, 224);
    return std::pair<Mat, Mat>(frame, depth);
}

std::pair<std::vector<KeyPoint>, Mat> Tracker::compute_features_descriptors(Mat frame) {
    std::vector<KeyPoint> keypoints;
    cv::Mat descriptors;
    fast->detect(frame, keypoints);
    orb->compute(frame, keypoints, descriptors);
    return std::pair<std::vector<KeyPoint>, cv::Mat>(keypoints, descriptors);
} 

Eigen::Vector3d Tracker::compute_camera_center(Eigen::Matrix4d camera_pose) {
    Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            R(i, j) = camera_pose(i, j);
        }
    }
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    for (int i = 0; i < 3; i++) {
        t(i) = camera_pose(3, i);
    }
    return -R.transpose() * t;
}

std::pair<Graph, vector<MapPoint>> Tracker::initialize(Eigen::Matrix3d K) {
    vector<MapPoint> map_points;
    std::pair<Mat, Mat> pair_frame_depth = get_next_image();
    Mat frame = pair_frame_depth.first;
    Mat depth_image = pair_frame_depth.second;
    std::pair<std::vector<KeyPoint>, Mat> pair_kp_des = compute_features_descriptors(frame);
    std::vector<cv::KeyPoint> kps = pair_kp_des.first;
    Mat des = pair_kp_des.second;
    Eigen::Matrix4d identity = Eigen::Matrix4d::Identity(); 
    Eigen::Vector3d camera_center = this->compute_camera_center(identity);
    for (int i = 0; i < kps.size(); i++) {
        float depth = depth_image.at<float>((int)kps[i].pt.x, (int)kps[i].pt.y);
        if (depth < 0.001) continue;
        map_points.push_back(MapPoint(kps[i], depth, identity, camera_center, des.row(i)));
    }
    this->last_frame = KeyFrame(identity, K, kps, des, depth_image);
    Graph graph = Graph(this->last_frame);
    return std::pair<Graph, vector<MapPoint>>(graph, map_points);
}


int Tracker::partition(vector<DMatch> &vec, int low, int high) {
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (vec[j].distance < vec[high].distance) {
            i++;
            swap(vec[i], vec[j]);
        }
    }
    swap(vec[i + 1], vec[high]);
    return (i + 1);
}

void Tracker::sort_matches_based_on_distance(vector<DMatch> &matches, int low, int high) {
    if (low < high) {
      int pi = partition(matches, low, high);
      sort_matches_based_on_distance(matches, low, pi - 1);
      sort_matches_based_on_distance(matches, pi + 1, high);
  }
}


void Tracker::estimatePose() {}


void Tracker::tracking() {
    std::pair<Mat, Mat> pair_frame_depth = get_next_image();
    Mat frame = pair_frame_depth.first;
    Mat depth_image = pair_frame_depth.second;
    std::pair<std::vector<KeyPoint>, Mat> pair_kp_des = compute_features_descriptors(frame);
    std::vector<cv::KeyPoint> kps = pair_kp_des.first;
    Mat des = pair_kp_des.second;
    vector<DMatch> matches;
    matcher->match(this->last_frame.orb_descriptors, des, matches);
    sort_matches_based_on_distance(matches, 0, matches.size() - 1);
    vector<DMatch> good_matches; 
    for (auto match : matches) {
        if(match.distance < LIMIT_MATCHING) {
            good_matches.push_back(match);
        }
    }
}