#include "../include/KeyFrame.h"

KeyFrame::KeyFrame() {};


std::vector<MapPoint*> KeyFrame::get_map_points() {
    std::vector<MapPoint*> mps;
    for (std::pair<MapPoint*, Feature*> it : this->mp_correlations) {
        mps.push_back(it.first);
    }
    return mps;
}

void KeyFrame::set_keyframe_position(Sophus::SE3d Tcw_new) {
    this->Tcw = Tcw_new;
    this->mat_camera_world = Tcw_new.matrix(); 
    this->camera_center_world = -this->Tcw.rotationMatrix().transpose() * this->Tcw.translation();
    Eigen::Quaterniond q =  this->Tcw.unit_quaternion();
    Eigen::Vector3d t = this->Tcw.translation();
    this->pose_vector[0] = q.w();
    this->pose_vector[1] = q.x();
    this->pose_vector[2] = q.y();
    this->pose_vector[3] = q.z();
    this->pose_vector[4] = t.x();
    this->pose_vector[5] = t.y();
    this->pose_vector[6] = t.z();
}


bool KeyFrame::check_map_point_in_keyframe(MapPoint *mp) {
    bool in_mp_correlations = this->mp_correlations.find(mp) != this->mp_correlations.end();
    bool in_features = in_mp_correlations ? this->mp_correlations[mp]->get_map_point() == mp : in_mp_correlations;
    if (in_mp_correlations && in_features) return true;
    if (!in_mp_correlations && !in_features) return false;
    if (in_mp_correlations != in_features) {
        std::cout << "NU S-A PUTUT NU ESTE SINCRONIZAT KEYFRAME-ul\n";
    }
    return false;
}

bool KeyFrame::check_number_close_points()
{
    int close_points_tracked = 0;
    int close_points_untracked = 0;
    for (Feature f : this->features) {
        if (f.is_monocular || f.depth >= 3.2) continue;
        if (f.get_map_point() == nullptr) close_points_untracked++;
        if (f.get_map_point() != nullptr) close_points_tracked++;
    }
    return (close_points_tracked < 100) && (close_points_untracked > 70);
}

void KeyFrame::create_feature_vector(std::vector<cv::KeyPoint> &keypoints, std::vector<cv::KeyPoint> &undistored_kps,
 cv::Mat orb_descriptors, cv::Mat depth_matrix) {
    for (int i = 0; i < (int)keypoints.size(); i++) { 
        cv::KeyPoint kp = keypoints[i];
        int x = std::round(keypoints[i].pt.x);
        int y = std::round(keypoints[i].pt.y);
        uint16_t d = depth_matrix.at<uint16_t>(y, x);
        float depth = d / 5000.0f;
        cv::KeyPoint kpu = undistored_kps[i];
        if (depth <= 1e-1)
        {
            this->features.push_back(Feature(kp, kpu, orb_descriptors.row(i), i, depth, 0));
        }
        else
        {
            double right_coordinate = kpu.pt.x - (this->K(0, 0) * BASELINE / depth);
            this->features.push_back(Feature(kp, kpu, orb_descriptors.row(i), i, depth, right_coordinate));
        }
    }
}

void KeyFrame::create_grid_matrix() {
    this->grid = std::vector<std::vector<std::vector<int>>>(10, std::vector<std::vector<int>>(10, std::vector<int>()));
    
    for (int i = 0; i < (int)this->features.size(); i++)
    {
        cv::KeyPoint kpu = this->features[i].get_undistorted_keypoint();
        int row = (int)kpu.pt.y / GRID_HEIGHT;
        int col = (int)kpu.pt.x / GRID_WIDTH;
        this->grid[row][col].push_back(i);
    }
}

KeyFrame::KeyFrame(KeyFrame* old_kf, std::vector<cv::KeyPoint> &keypoints,
    std::vector<cv::KeyPoint> &undistored_kps, cv::Mat orb_descriptors, cv::Mat depth_matrix) : K(old_kf->K), voc(old_kf->voc), reference_kf(old_kf->reference_kf), reference_idx(old_kf->reference_idx) {
    this->set_keyframe_position(old_kf->Tcw);
    this->current_idx = old_kf->current_idx + 1;
    this->create_feature_vector(keypoints, undistored_kps, orb_descriptors, depth_matrix);        
    this->create_grid_matrix();
    this->minX = old_kf->minX;
    this->maxX = old_kf->maxX;
    this->minY = old_kf->minY;
    this->maxY = old_kf->maxY;
    this->mp_correlations.reserve(this->features.size() * 1.3); 
}

KeyFrame::KeyFrame(Sophus::SE3d Tcw, Eigen::Matrix3d K, std::vector<double> distorsion, std::vector<cv::KeyPoint> &keypoints,
     std::vector<cv::KeyPoint> &undistored_kps, cv::Mat orb_descriptors, cv::Mat depth_matrix, int current_idx, 
     cv::Mat &frame, ORBVocabulary *voc) : K(K), current_idx(current_idx), voc(voc), reference_kf(nullptr), reference_idx(-1)
{
    this->set_keyframe_position(Tcw);
    this->create_feature_vector(keypoints, undistored_kps, orb_descriptors, depth_matrix);
    this->create_grid_matrix();

    if (distorsion[0] != 0.0)
    {
        cv::Mat mat(4, 2, CV_32F);
        mat.at<float>(0, 0) = 0.0;
        mat.at<float>(0, 1) = 0.0;
        mat.at<float>(1, 0) = frame.cols;
        mat.at<float>(1, 1) = 0.0;
        mat.at<float>(2, 0) = 0.0;
        mat.at<float>(2, 1) = frame.rows;
        mat.at<float>(3, 0) = frame.cols;
        mat.at<float>(3, 1) = frame.rows;

        // Undistort corners
        mat = mat.reshape(2);
        cv::Mat K_cv;
        cv::eigen2cv(K, K_cv);
        cv::undistortPoints(mat, mat, K_cv, distorsion, cv::Mat(), K_cv);
        mat = mat.reshape(1);

        this->minX = std::min(mat.at<float>(0, 0), mat.at<float>(2, 0));
        this->maxX = std::max(mat.at<float>(1, 0), mat.at<float>(3, 0));
        this->minY = std::min(mat.at<float>(0, 1), mat.at<float>(1, 1));
        this->maxY = std::max(mat.at<float>(2, 1), mat.at<float>(3, 1));
    }
    else
    {
        this->minX = 0.0f;
        this->maxX = frame.cols;
        this->minY = 0.0f;
        this->maxY = frame.rows;
    }
    this->mp_correlations.reserve(this->features.size() * 1.3);
}


void KeyFrame::set_reference_keyframe(KeyFrame *ref) {
    this->reference_kf = ref;
    this->reference_idx = ref->current_idx;
}

void KeyFrame::compute_bow_representation()
{
    if (this->bow_vec.size() != 0 && this->features_vec.size() != 0) return;
    std::vector<cv::Mat> vector_descriptors;
    for (int i = 0; i < (int)this->features.size(); i++)
    {
        vector_descriptors.push_back(this->features[i].descriptor);
    }
    this->voc->transform(vector_descriptors, this->bow_vec, this->features_vec, 4);
}

Sophus::SE3d KeyFrame::compute_pose() {
    Eigen::Quaterniond quaternion(this->pose_vector[0], this->pose_vector[1], this->pose_vector[2], this->pose_vector[3]);
    Eigen::Quaterniond old_quaternion = this->Tcw.unit_quaternion();
    if (old_quaternion.dot(quaternion) < 0)
    {
        quaternion.coeffs() *= -1;
    }
    return Sophus::SE3d(quaternion, Eigen::Vector3d(this->pose_vector[4], this->pose_vector[5], this->pose_vector[6]));
}

int KeyFrame::get_map_points_seen_from_multiple_frames(int nr_frames) {
    int out = 0;
    for (Feature f : this->features) {
        MapPoint *mp = f.get_map_point();
        if (mp == nullptr) continue;
        if ((int)mp->data.size() >= nr_frames) out++;
    }
    return out;
}

std::vector<cv::KeyPoint> KeyFrame::get_all_keypoints()
{
    std::vector<cv::KeyPoint> out;
    for (Feature f : this->features)
    {
        out.push_back(f.get_key_point());
    }
    return out;
}

std::vector<int> KeyFrame::get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave)
{
    const int min_bin_u = std::max(u - window, this->minX) / GRID_WIDTH;
    const int max_bin_u = std::min(u + window, this->maxX - 1) / GRID_WIDTH;
    const int min_bin_v = std::max(v - window, this->minY) / GRID_HEIGHT;
    const int max_bin_v = std::min(v + window, this->maxY - 1) / GRID_HEIGHT;
    std::vector<int> kps_idx;
    kps_idx.reserve((max_bin_u - min_bin_u + 1) * (max_bin_v - min_bin_v + 1) * 5);
    for (int i = min_bin_u; i <= max_bin_u; i++)
    {
        for (int j = min_bin_v; j <= max_bin_v; j++)
        {
            if (this->grid[j][i].empty()) continue;
            for (int idx : this->grid[j][i]) {
                const cv::KeyPoint kpu = this->features[idx].kpu;
                if (kpu.octave < minOctave || kpu.octave > maxOctave)
                    continue;
                const float xdist = kpu.pt.x - u;
                const float ydist = kpu.pt.y - v;
                if (fabs(xdist) > window || fabs(ydist) > window)
                    continue;
                kps_idx.push_back(idx);
            }       
        }
    }
    return kps_idx;
}


void KeyFrame::debug_keyframe(cv::Mat frame, int miliseconds)
{
    std::vector<cv::KeyPoint> keypoints;
    for (auto it =this->mp_correlations.begin(); it != this->mp_correlations.end(); it++)
    {
        keypoints.push_back(it->second->get_key_point());
    }

    cv::Mat img2, img3;
    cv::drawKeypoints(frame, this->get_all_keypoints(), img2, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
    // cv::imshow("Display window", img2);
    // cv::waitKey(0);
    cv::drawKeypoints(frame, keypoints, img3, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Display window", img3);
    cv::waitKey(miliseconds);
}


bool KeyFrame::debug_keyframe_valid() {
    for (std::pair<MapPoint*, Feature*> it : this->mp_correlations) {
        MapPoint *mp = it.first;
        Feature *f = it.second;
        if (f->get_map_point() != mp) return false;
    }
    return true;
}