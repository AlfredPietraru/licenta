#include "../include/LocalMap.h"

LocalMap::LocalMap(Graph &graph, Map &mapp) : graph(graph), mapp(mapp) {}

int LocalMap::check_common_map_points(KeyFrame *keyframe_with_map_points, KeyFrame *simple_keyframe) 
{
    std::vector<MapPoint*> relevant_map_points = this->mapp.get_map_points(keyframe_with_map_points);
    int res = 0; 
    for (MapPoint *mp : relevant_map_points) {
        res += keyframe_with_map_points->map_point_belongs_to_keyframe(mp);
    }
    return res;
}



void LocalMap::Add_New_KeyFrame(std::vector<KeyFrame*> &keyframes_buffer) {
    if (keyframes_buffer.size() == 0) return;
    KeyFrame *keyframe = keyframes_buffer.back();
    keyframes_buffer.pop_back();
}