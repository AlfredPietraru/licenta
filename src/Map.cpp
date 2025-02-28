#include "../include/Map.h"

Map::Map() {}

std::vector<MapPoint *> Map::get_map_points(KeyFrame *frame)
{
    if (map_points.find(frame) == map_points.end())
        return {};
    return map_points[frame];
}

void Map::add_map_points(KeyFrame *frame) {
    std::vector<MapPoint *> current_points_found;
    Eigen::Vector3d camera_center = compute_camera_center(frame->Tiw);
    for (cv::KeyPoint kp : frame->keypoints) {
        double depth = frame->depth_matrix.at<float>((int)kp.pt.x, (int)kp.pt.y);
        if (depth <= 0) continue;
        current_points_found.push_back(new MapPoint(frame, kp, depth, frame->Tiw, camera_center, frame->orb_descriptors));
    }
    if (current_points_found.size() == 0) return;
    std::cout << current_points_found.size() << "\n";
    this->map_points.insert(std::pair<KeyFrame*, std::vector<MapPoint*>>(frame, current_points_found));
}


int Map::check_common_map_points(KeyFrame *kf1, KeyFrame *kf2)
{
    std::vector<MapPoint *> first_kf_map_points = this->get_map_points(kf1);
    std::vector<MapPoint *> second_kf_map_points = this->get_map_points(kf2);
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