#include "../include/KeyFrame.h"

KeyFrame::KeyFrame() {};

std::unordered_set<MapPoint *> KeyFrame::return_map_points_frame()
{
    return this->map_points;
}

void KeyFrame::add_outlier_element(MapPoint *mp)
{
    this->outliers.insert(mp);
}

void KeyFrame::remove_outlier_element(MapPoint *mp)
{
    if (this->outliers.find(mp) != this->outliers.end())
    {
        this->outliers.erase(mp);
    }
}

bool KeyFrame::check_map_point_outlier(MapPoint *mp)
{
    if (this->outliers.find(mp) != this->outliers.end())
        return true;
    return false;
}


int KeyFrame::check_number_close_points()
{
    int close_points = 0;
    for (auto it = this->mp_correlations.begin(); it != this->mp_correlations.end(); it++) {
        Feature *f = it->second;
        if (f->depth < 3.2 && f->depth > 1e-6) close_points++;
    }
    return close_points;
}

KeyFrame::KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<double> distorsion, std::vector<cv::KeyPoint> &keypoints,
     std::vector<cv::KeyPoint> &undistored_kps, cv::Mat orb_descriptors, cv::Mat depth_matrix,
      int current_idx, cv::Mat &frame, ORBVocabulary *voc, KeyFrame *reference_kf)
    : Tiw(Tiw), K(K), orb_descriptors(orb_descriptors), current_idx(current_idx), frame(frame), voc(voc), reference_kf(reference_kf)
{

    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::KeyPoint kp = keypoints[i];
        int x = std::round(kp.pt.x);
        int y = std::round(kp.pt.y);
        uint16_t d = depth_matrix.at<uint16_t>(y, x);
        float depth = d / 5000.0f;
        cv::KeyPoint kpu = undistored_kps[i];
        if (depth <= 1e-6)
        {
            this->features.push_back(Feature(kp, kpu, orb_descriptors.row(i), i, -10001, -10001));
        }
        else
        {
            double rgbd_right_coordinate = kpu.pt.x - (this->K(0, 0) * BASELINE / depth);
            this->features.push_back(Feature(kp, kpu, orb_descriptors.row(i), i, depth, rgbd_right_coordinate));
        }
    }

    this->grid = std::vector<std::vector<std::vector<int>>>(10, std::vector<std::vector<int>>(10, std::vector<int>()));
    
    for (int i = 0; i < this->features.size(); i++)
    {
        cv::KeyPoint kpu = this->features[i].get_undistorted_keypoint();
        int y_idx = (int)kpu.pt.y;
        int x_idx = (int)kpu.pt.x;
        this->grid[y_idx / GRID_WIDTH][x_idx / GRID_HEIGHT].push_back(i);
    }

    // for (int i = 0; i < 10; i++) {
    //     for (int j = 0; j < 10; j++) {
    //         std::cout << this->grid[i][j].size() <<  " " << i * GRID_WIDTH << " " << j * GRID_HEIGHT << " dimensiune initiala\n";
    //         for (int k = 0; k < this->grid[i][j].size(); k++) {
    //             std::cout << this->features[this->grid[i][j][k]].kpu.pt.y  << " " << this->features[this->grid[i][j][k]].kpu.pt.x << "     ";
    //         }
    //         std::cout << "\n\n";
    //     }
    // }

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
}

void KeyFrame::compute_bow_representation()
{
    std::vector<cv::Mat> vector_descriptors;
    for (int i = 0; i < this->orb_descriptors.rows; i++)
    {
        vector_descriptors.push_back(this->orb_descriptors.row(i));
    }
    this->voc->transform(vector_descriptors, this->bow_vec, this->features_vec, 4);
}

Eigen::Vector3d KeyFrame::compute_camera_center()
{
    return -this->Tiw.rotationMatrix().transpose() * this->Tiw.translation();
}

Eigen::Vector3d KeyFrame::fromWorldToImage(Eigen::Vector4d &wcoord)
{
    Eigen::Vector4d camera_coordinates = this->Tiw.matrix() * wcoord;
    double d = camera_coordinates(2);
    double u = this->K(0, 0) * camera_coordinates(0) / d + this->K(0, 2);
    double v = this->K(1, 1) * camera_coordinates(1) / d + this->K(1, 2);
    return Eigen::Vector3d(u, v, d);
}

Eigen::Vector4d KeyFrame::fromImageToWorld(int kp_idx)
{
    Feature f = this->features[kp_idx];
    double new_x = (f.kp.pt.x - this->K(0, 2)) * f.depth / this->K(0, 0);
    double new_y = (f.kp.pt.y - this->K(1, 2)) * f.depth / this->K(1, 1);
    return this->Tiw.inverse().matrix() * Eigen::Vector4d(new_x, new_y, f.depth, 1);
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

void KeyFrame::add_map_point(MapPoint *mp, Feature *f) {
    if (this->outliers.find(mp) != this->outliers.end()) {
        this->outliers.erase(mp);
    }
    this->map_points.insert(mp);
    this->mp_correlations.insert({mp, f});
    mp->increase_number_associations();
}

void KeyFrame::remove_map_point(MapPoint *mp) {
    if(this->mp_correlations.find(mp) != this->mp_correlations.end()) {
        Feature *f = this->mp_correlations[mp];
        // era functia asta stearsa\n
        // f->unmatch_map_point();
        this->mp_correlations.erase(mp);
    }
    if (this->map_points.find(mp) != this->map_points.end()) this->map_points.erase(mp);
    mp->decrease_number_associations();
}


void KeyFrame::update_map_points_info() {
    for (MapPoint *mp : this->map_points) {
        
    }
}


void KeyFrame::correlate_map_points_to_features_current_frame(std::unordered_map<MapPoint *, Feature *>& matches)
{
    int e_bine = 0;
    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        Feature *f = it->second;
        MapPoint *mp = it->first;
        MapPoint *old_mp = f->get_map_point();
        bool old_map_point_was_replaced = f->set_map_point(mp);
        if (old_map_point_was_replaced && old_mp == nullptr) {
            this->add_map_point(mp, f);
        }
        if (old_map_point_was_replaced && old_mp != nullptr) {
            this->remove_map_point(old_mp);
            this->add_map_point(mp, f);
        }
    }
}

std::vector<int> KeyFrame::get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave)
{
    int u_min = lround(u - window);
    u_min = (u_min < 0) ? 0 : (u_min >= this->frame.cols) ? this->frame.cols - 1
    : u_min;
    int u_max = lround(u + window);
    u_max = (u_max < 0) ? 0 : (u_max >= this->frame.cols) ? this->frame.cols - 1
    : u_max;
    int v_min = round(v - window);
    v_min = (v_min < 0) ? 0 : (v_min >= this->frame.rows) ? this->frame.rows - 1
    : v_min;
    int v_max = lround(v + window);
    v_max = (v_max < 0) ? 0 : (v_max >= this->frame.rows) ? this->frame.rows - 1
    : v_max;
    
    std::vector<int> kps_idx;
    int min_bin_u = u_min / GRID_HEIGHT;
    int max_bin_u = u_max / GRID_HEIGHT;
    int min_bin_v = v_min / GRID_WIDTH;
    int max_bin_v = v_max / GRID_WIDTH;
    for (int i = min_bin_u; i <= max_bin_u; i++)
    {
        for (int j = min_bin_v; j <= max_bin_v; j++)
        {
            std::vector<int> current_feature_idxs = this->grid[j][i];
            for (int idx : current_feature_idxs) {
                cv::KeyPoint kpu = this->features[idx].kpu;
                if (kpu.octave < minOctave || kpu.octave > maxOctave)
                    continue;
                float xdist = kpu.pt.x - u;
                float ydist = kpu.pt.y - v;
                if (fabs(xdist) > window || fabs(ydist) > window)
                    continue;
                kps_idx.push_back(idx);
            }       
        }
    }
    return kps_idx;
}


void KeyFrame::debug_keyframe(int miliseconds, std::unordered_map<MapPoint *, Feature *> &matches, std::unordered_map<MapPoint *, Feature *> &new_matches)
{
    std::vector<cv::KeyPoint> keypoints;
    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        new_matches.insert({it->first, it->second});
    }
    for (auto it = new_matches.begin(); it != new_matches.end(); it++)
    {
        keypoints.push_back(it->second->get_key_point());
    }

    cv::Mat img2, img3;
    cv::drawKeypoints(this->frame, this->get_all_keypoints(), img2, cv::Scalar(255, 0, 0), cv::DrawMatchesFlags::DEFAULT);
    // cv::imshow("Display window", img2);
    // cv::waitKey(0);
    cv::drawKeypoints(this->frame, keypoints, img3, cv::Scalar(0, 0, 255), cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("Display window", img3);
    cv::waitKey(miliseconds);
}
