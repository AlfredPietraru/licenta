#include "Map.h"

Map::Map() {}

void Map::add_multiple_map_points(KeyFrame frame, std::vector<MapPoint*> new_points)
{
    for (int i = 0; i < new_points.size(); i++)
    {
        map_points.push_back(new_points[i]);
    }
}

std::vector<MapPoint*> Map::get_map_points(KeyFrame frame)
{
    return map_points;
}

std::vector<MapPoint*> Map::return_map_points_seen_in_frame(Eigen::Matrix4d &pose, cv::Mat &depth)
{
    std::vector<MapPoint*> out;
    for (MapPoint *mp : this->map_points)
    {
        std::pair<float, float> camera_coordinates = fromWorldToCamera(pose, depth, mp->wcoord);
        if (camera_coordinates.first < 0 || camera_coordinates.second < 0)
            continue;
        out.push_back(mp);
    }
    return out;
}
