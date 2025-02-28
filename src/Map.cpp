#include "../include/Map.h"

Map::Map() {}

std::vector<MapPoint *> Map::get_map_points(KeyFrame *curr_frame, KeyFrame *reference_kf)
{
    std::vector<MapPoint*> reference_map_points = map_points[reference_kf];
    if ( reference_map_points.size() == 0) return {};
    std::vector<MapPoint*> out;
    for (MapPoint *mp : reference_map_points)
    {
        if (mp->map_point_belongs_to_keyframe(curr_frame)) {
            out.push_back(mp);
        }
    }
    return out;
}

void Map::add_map_points(KeyFrame *frame) {
    std::vector<MapPoint *> current_points_found;
    Eigen::Vector3d camera_center = frame->compute_camera_center();
    for (int i = 0; i < frame->keypoints.size(); i++) {
        cv::KeyPoint kp = frame->keypoints[i];
        double depth = frame->depth_matrix.at<float>((int)kp.pt.x, (int)kp.pt.y);
        if (depth <= 0) continue;
        current_points_found.push_back(new MapPoint(frame, i));
    }
    if (current_points_found.size() == 0) return;
    this->map_points.insert(std::pair<KeyFrame*, std::vector<MapPoint*>>(frame, current_points_found));
}

// DE MODIFICAT NU E BINE
int Map::check_common_map_points(KeyFrame *kf1, KeyFrame *kf2)
{
    std::vector<MapPoint *> first_kf_map_points = this->get_map_points(kf1, kf2);
    std::vector<MapPoint *> second_kf_map_points = this->get_map_points(kf2, kf1);
    int res = 0;
    if (first_kf_map_points.size() == 0 && second_kf_map_points.size() == 0) return 0;
    if (first_kf_map_points.size() != 0 && second_kf_map_points.size() == 0)
    {
        for (MapPoint *mp : first_kf_map_points)
        {
            res += mp->map_point_belongs_to_keyframe(kf2);
        }
        return res;
    }
    if (first_kf_map_points.size() == 0 && second_kf_map_points.size() != 0)
    {
        for (MapPoint *mp : second_kf_map_points)
        {
            res += mp->map_point_belongs_to_keyframe(kf1);
        }
        return res;
    }
    std::unordered_set<MapPoint *> first_kf_map_points_seen_from_second;
    std::unordered_set<MapPoint *> second_kf_map_points_seen_from_second;
    for (MapPoint *mp : first_kf_map_points)
    {
        if (mp->map_point_belongs_to_keyframe(kf2))
            first_kf_map_points_seen_from_second.insert(mp);
    }
    for (MapPoint *mp : second_kf_map_points)
    {
        if (mp->map_point_belongs_to_keyframe(kf1))
            second_kf_map_points_seen_from_second.insert(mp);
    }
    first_kf_map_points_seen_from_second.insert(second_kf_map_points_seen_from_second.begin(),
                                                second_kf_map_points_seen_from_second.end());
    return first_kf_map_points_seen_from_second.size();
}