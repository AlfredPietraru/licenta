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

cv::Mat Tracker::convert_from_eigen_to_cv2(Eigen::MatrixX<double> matrix) {
    cv::Mat out = cv::Mat::zeros(cv::Size(matrix.rows(), matrix.cols()), CV_64FC1);
    for (int i = 0; i < matrix.rows(); i++) {
        for (int j = 0; j < matrix.cols(); j++) {
            out.at<double>(i,j) = matrix(i, j);
        }
    }
    return out;
}

Eigen::Matrix4d Tracker::estimate_pose(KeyFrame frame, std::vector<DMatch> matches, std::vector<KeyPoint> keypoints_2, Mat depth, Eigen::Matrix3d K) {
    vector<Point3d> points_in3d;
    vector<Point2d> points_in2d;
    cout << matches.size() << "\n";
    for (DMatch m : matches) {
      float d = frame.depth_matrix.ptr<float>(int(frame.keypoints[m.queryIdx].pt.y))[int(frame.keypoints[m.queryIdx].pt.x)];
      if (d <= 0.01) continue;
      float new_x = (frame.keypoints[m.queryIdx].pt.x - X_CAMERA_OFFSET) * d / FOCAL_LENGTH;
      float new_y = (frame.keypoints[m.queryIdx].pt.y - Y_CAMERA_OFFSET) * d / FOCAL_LENGTH;
      points_in3d.push_back(Point3d(new_x, new_y, d));
      points_in2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    cout << points_in2d.size() << " " << points_in3d.size() << "\n";
    Mat r, t;
    // pag 160
    cv::solvePnP(points_in3d, points_in2d, convert_from_eigen_to_cv2(K), Mat(), r, t, false);
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
    return T;
}


void Tracker::tracking(Eigen::Matrix3d K) {
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
    cout << estimate_pose(this->last_frame, good_matches, kps, depth_image, K);

}