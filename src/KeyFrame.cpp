#include "../include/KeyFrame.h"

KeyFrame::KeyFrame() {};

std::unordered_set<MapPoint *> KeyFrame::return_map_points_frame()
{
    return this->map_points;
}

std::unordered_map<MapPoint *, Feature *> KeyFrame::return_map_points_keypoint_correlation()
{
    std::unordered_map<MapPoint *, Feature *> out;
    for (int i = 0; i < this->features.size(); i++)
    {
        MapPoint *mp = this->features[i].get_map_point();
        if (mp == nullptr)
            continue;
        out.insert({mp, &this->features[i]});
    }
    return out;
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

int KeyFrame::check_possible_close_points_generation()
{
    return this->outliers.size();
}

int KeyFrame::check_number_close_points()
{
    return this->map_points.size();
}

KeyFrame::KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<double> distorsion, std::vector<cv::KeyPoint> &keypoints, std::vector<cv::KeyPoint> &undistored_kps,
                   cv::Mat orb_descriptors, cv::Mat depth_matrix, int current_idx, cv::Mat &frame, ORBVocabulary *voc)
    : Tiw(Tiw), K(K), orb_descriptors(orb_descriptors), depth_matrix(depth_matrix), current_idx(current_idx), frame(frame)
{

    for (int i = 0; i < keypoints.size(); i++)
    {
        cv::KeyPoint kp = keypoints[i];
        cv::KeyPoint kpu = undistored_kps[i];
        double depth = this->compute_depth_in_keypoint(kp);
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

    this->grid = cv::Mat::zeros(frame.rows, frame.cols, CV_32S);
    this->grid += cv::Scalar(-1);

    for (int i = 0; i < this->features.size(); i++)
    {
        Feature f = this->features[i];
        int y_idx = (int)f.kpu.pt.y;
        int x_idx = (int)f.kpu.pt.x;
        if (this->grid.at<int>(y_idx, x_idx) == -1)
        {
            this->grid.at<int>(y_idx, x_idx) = i;
            continue;
        }
        if (f.kpu.pt.y - y_idx > 0.5 && this->grid.at<int>(y_idx + 1, x_idx) == -1)
        {
            this->grid.at<int>(y_idx + 1, x_idx) == i;
            continue;
        }
        if (f.kpu.pt.x - x_idx > 0.5 && this->grid.at<int>(y_idx, x_idx + 1) == -1)
        {
            this->grid.at<int>(y_idx, x_idx + 1) == i;
            continue;
        }
        this->grid.at<int>(y_idx + 1, x_idx + 1) = i;
    }

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

    this->compute_bow_representation(voc);
}

void KeyFrame::compute_bow_representation(ORBVocabulary *voc)
{
    std::vector<cv::Mat> vector_descriptors;
    for (int i = 0; i < this->orb_descriptors.rows; i++)
    {
        vector_descriptors.push_back(this->orb_descriptors.row(i));
    }
    voc->transform(vector_descriptors, this->bow_vec, this->features_vec, 4);
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

float KeyFrame::compute_depth_in_keypoint(cv::KeyPoint kp)
{
    int x = std::round(kp.pt.x);
    int y = std::round(kp.pt.y);
    uint16_t d = this->depth_matrix.at<uint16_t>(y, x);
    float dd = d / 5000.0f;
    return dd;
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

Eigen::Vector3d KeyFrame::get_viewing_direction()
{
    return Tiw.rotationMatrix().col(2).normalized();
}

void KeyFrame::correlate_map_points_to_features_current_frame(std::unordered_map<MapPoint *, Feature *> &matches)
{
    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        MapPoint *mp = it->first;
        if (this->check_map_point_outlier(mp)) continue;
        it->second->set_map_point(mp);
        this->map_points.insert(mp);
    }
}

std::vector<int> KeyFrame::get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave)
{
    std::vector<int> kps_idx;
    int u_min = lround(u - window);
    u_min = (u_min < 0) ? 0 : (u_min > this->frame.cols) ? this->frame.cols
                                                         : u_min;
    int u_max = lround(u + window);
    u_max = (u_max < 0) ? 0 : (u_max > this->frame.cols) ? this->frame.cols
                                                         : u_max;
    int v_min = round(v - window);
    v_min = (v_min < 0) ? 0 : (v_min > this->frame.rows) ? this->frame.rows
                                                         : v_min;
    int v_max = lround(v + window);
    v_max = (v_max < 0) ? 0 : (v_max > this->frame.rows) ? this->frame.rows
                                                         : v_max;
    for (int i = v_min; i < v_max; i++)
    {
        for (int j = u_min; j < u_max; j++)
        {
            if (this->grid.at<int>(i, j) == -1)
                continue;
            cv::KeyPoint kpu = this->features[this->grid.at<int>(i, j)].kpu;
            int current_octave = kpu.octave;
            if (current_octave < minOctave || current_octave > maxOctave)
                continue;
            float xdist = kpu.pt.x - u;
            float ydist = kpu.pt.y - v;
            if (fabs(xdist) > window || fabs(ydist) > window)
                continue;
            kps_idx.push_back(this->grid.at<int>(i, j));
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
