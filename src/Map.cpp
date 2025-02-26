#include "../include/Map.h"

Map::Map() {}

void Map::add_multiple_map_points(KeyFrame *frame, std::vector<MapPoint *> new_points)
{
    this->map_points.insert(std::pair<KeyFrame *, std::vector<MapPoint *>>(frame, new_points));
}

std::vector<MapPoint *> Map::get_map_points(KeyFrame *frame)
{
    if (map_points.find(frame) == map_points.end())
        return {};
    return map_points[frame];
}

std::vector<MapPoint *> Map::return_map_points_seen_in_frame(KeyFrame *frame, Eigen::Matrix4d &pose, cv::Mat &depth)
{
    std::vector<MapPoint *> out;
    for (MapPoint *mp : this->map_points[frame])
    {
        std::pair<float, float> camera_coordinates = fromWorldToCamera(pose, depth, mp->wcoord);
        if (camera_coordinates.first < 0 || camera_coordinates.second < 0)
            continue;
        out.push_back(mp);
    }
    return out;
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