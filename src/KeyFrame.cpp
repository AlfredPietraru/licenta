#include "../include/KeyFrame.h"

KeyFrame::KeyFrame(){};

std::unordered_set<MapPoint*> KeyFrame::return_map_points_frame() {
    return this->map_points;
}

std::unordered_map<MapPoint*, Feature*> KeyFrame::return_map_points_keypoint_correlation() {
    std::unordered_map<MapPoint*, Feature*> out;
    for (int i = 0; i < this->features.size(); i++) {
        MapPoint *mp = this->features[i].get_map_point();
        if (mp == nullptr) continue;
        out.insert({mp, &this->features[i]});
    }
    return out;
}

void KeyFrame::add_outlier_element(MapPoint *mp) {
    this->outliers.insert(mp);
}
bool KeyFrame::check_map_point_outlier(MapPoint *mp) {
    if (this->outliers.find(mp) != this->outliers.end()) return true;
    return false;
 }


KeyFrame::KeyFrame(Sophus::SE3d Tiw, Eigen::Matrix3d K, std::vector<cv::KeyPoint>& keypoints,
         cv::Mat orb_descriptors, cv::Mat depth_matrix, int idx, cv::Mat& frame, ORBVocabulary *voc)
    : Tiw(Tiw), K(K), orb_descriptors(orb_descriptors), depth_matrix(depth_matrix), idx(idx), frame(frame) {
        this->grid = cv::Mat::zeros(frame.rows, frame.cols, CV_32S);
        this->grid += cv::Scalar(-1);
        this->maximum_possible_map_points = keypoints.size();
        for (int i = 0; i < keypoints.size(); i++) {
            cv::KeyPoint kp = keypoints[i];
            double depth = this->compute_depth_in_keypoint(kp);
            if(depth <= 1e-6) {
                this->maximum_possible_map_points--;
                this->features.push_back(Feature(kp, orb_descriptors.row(i), i, -1)); // filler negative value for stereo depth
            } else { 
                double rgbd_right_coordinate = kp.pt.x - (this->K(0, 0) * BASELINE / depth);
                this->features.push_back(Feature(kp, orb_descriptors.row(i), i, rgbd_right_coordinate));
            }
            this->grid.at<int>(lround(kp.pt.y), lround(kp.pt.x)) = i;
        }

        std::vector<cv::Mat> vector_descriptors;
        for (int i = 0; i < this->orb_descriptors.rows; i++) {
            vector_descriptors.push_back(this->orb_descriptors.row(i));
        }
        voc->transform(vector_descriptors, this->bow_vec, this->features_vec, 3);
    }

Eigen::Vector3d KeyFrame::compute_camera_center() {
    return -this->Tiw.rotationMatrix().transpose() * this->Tiw.translation();
}

Eigen::Vector3d KeyFrame::fromWorldToImage(Eigen::Vector4d& wcoord) {
    Eigen::Vector4d camera_coordinates = this->Tiw.matrix() * wcoord;
    double d = camera_coordinates(2);
    double u = this->K(0, 0) * camera_coordinates(0) / d + this->K(0, 2);
    double v = this->K(1, 1) * camera_coordinates(1) / d + this->K(1, 2);
    double stereo_depth = u - (this->K(0, 0) * BASELINE / d);
    return Eigen::Vector3d(u, v, stereo_depth);
}


float KeyFrame::compute_depth_in_keypoint(cv::KeyPoint kp) {
    int x = std::round(kp.pt.x);
    int y = std::round(kp.pt.y);
    uint16_t d = this->depth_matrix.at<uint16_t>(y, x);
    float dd = d / 5000.0f;
    return dd;
}

Eigen::Vector4d KeyFrame::fromImageToWorld(int kp_idx) {
    cv::KeyPoint kp = this->features[kp_idx].kp;
    float dd = this->compute_depth_in_keypoint(kp);
    double new_x = (kp.pt.x - this->K(0, 2)) * dd / this->K(0, 0);
    double new_y = (kp.pt.y - this->K(1, 2)) * dd / this->K(1, 1);
    return this->Tiw.inverse().matrix() *  Eigen::Vector4d(new_x, new_y, dd, 1);
}

cv::KeyPoint KeyFrame::get_keypoint(int idx) {
    return this->features[idx].get_key_point();
}


std::vector<cv::KeyPoint> KeyFrame::get_all_keypoints() {
    std::vector<cv::KeyPoint> out;
    for (Feature f : this->features) {
        out.push_back(f.get_key_point());
    }
    return out;
}

Eigen::Vector3d KeyFrame::get_viewing_direction() {
    return Tiw.rotationMatrix().col(2).normalized(); 
}

void KeyFrame::correlate_map_points_to_features_current_frame(std::unordered_map<MapPoint *, Feature*>& matches) {
    for (auto it = matches.begin(); it != matches.end(); it++) {
        it->second->set_map_point(it->first);
        this->map_points.insert(it->first);
    }
}

std::vector<int> KeyFrame::get_vector_keypoints_after_reprojection(double u, double v, int window, int minOctave, int maxOctave) {
    std::vector<int> kps_idx;
    int u_min = lround(u - window);
    u_min = (u_min < 0) ? 0 : (u_min > this->frame.cols) ? this->frame.cols : u_min;   
    int u_max = lround(u + window);
    u_max = (u_max < 0) ? 0 : (u_max > this->frame.cols) ? this->frame.cols : u_max;
    int v_min = round(v - window);
    v_min = (v_min < 0) ? 0 : (v_min > this->frame.rows) ? this->frame.rows : v_min;
    int v_max = lround(v + window);
    v_max = (v_max < 0) ? 0 : (v_max > this->frame.rows) ? this->frame.rows : v_max;
    for (int i = v_min; i < v_max; i++) {
        for (int j = u_min; j < u_max; j++) {
            if (this->grid.at<int>(i, j) == -1) continue;
            cv::KeyPoint kp = this->features[this->grid.at<int>(i, j)].kp;
            int current_octave = kp.octave;
            if (current_octave < minOctave || current_octave > maxOctave) continue;
            float xdist = kp.pt.x - u;
            float ydist = kp.pt.y - v;
            if(fabs(xdist) > window || fabs(ydist) > window) continue;
            kps_idx.push_back(this->grid.at<int>(i, j));
        }
    }
    return kps_idx;
}

void KeyFrame::compute_map_points()
{
    int map_points_associated = 0;
    int negative_depth = 0;
    int close_map_points = 0;
    int far_map_poins = 0;
    Eigen::Vector3d camera_center = this->compute_camera_center();
    for (int i = 0; i < this->features.size(); i++)
    {
        if (this->features[i].get_map_point() != nullptr) {
            map_points_associated++;
            continue;
        }
        double depth = this->compute_depth_in_keypoint(this->features[i].kp);
        if (depth <= 1e-6) {
            negative_depth++;
            continue;  
        } 
        Eigen::Vector4d wcoord = this->fromImageToWorld(i);
        MapPoint *mp = new MapPoint(this, this->features[i].kp, camera_center, wcoord,  this->orb_descriptors.row(i), i);
        this->features[i].set_map_point(mp);
        this->map_points.insert(mp);
        // if (mp->is_safe_to_use) close_map_points++;
        // if (!mp->is_safe_to_use) far_map_poins++;
    }
    if (this->map_points.size() == 0) {
        std::cout << "CEVA NU E BINE NU S-AU CREAT PUNCTELE\n";
    }
    std::cout << this->map_points.size() << " " << this->features.size() << " " << map_points_associated << " " << negative_depth << " debug compute map points\n";  
}



