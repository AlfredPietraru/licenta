#include "../include/MapPoint.h"


int MapPoint::ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b) {
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }
    return dist;
}

bool MapPoint::map_point_should_be_deleted() {
    double ratio = (double)this->number_associations / this->number_times_seen;
    return ratio < 0.25f;
}


MapPoint::MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, 
        cv::Mat orb_descriptor) : wcoord(wcoord)
{
    this->wcoord_3d = Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
    this->octave = kp.octave;
    this->add_observation(keyframe, camera_center, orb_descriptor);   
}

bool MapPoint::find_keyframe(KeyFrame *kf) {
    return this->data.find(kf) != this->data.end();
}

void MapPoint::add_observation(KeyFrame *kf, Eigen::Vector3d camera_center, cv::Mat orb_descriptor) {
    this->data.insert({kf, new MapPointEntry(-1, camera_center, orb_descriptor)});
    this->keyframes.push_back(kf);
    this->compute_distinctive_descriptor();
    this->compute_view_direction();
    this->compute_distance();
};

void MapPoint::remove_observation(KeyFrame *kf) {
    if (this->data.find(kf) == this->data.end()) return;
    this->data.erase(kf);
    for (int i = 0; i < (int)this->keyframes.size(); i++) {
        if (this->keyframes[i] == kf) {
            this->keyframes.erase(this->keyframes.begin() + i);
        }
    }
    this->compute_distinctive_descriptor();
    this->compute_view_direction();
    this->compute_distance();
}

void MapPoint::increase_how_many_times_seen() {
    this->number_times_seen += 1;
}

void MapPoint::increase_number_associations(int val) {
    this->number_associations += val;
}

void MapPoint::decrease_number_associations(int val) {
    if (this->number_associations == 0) {
        std::cout << "NU POATE FI SCAZUT MAI TARE NUMARUL DE ASOCIERI DE 0\n";
    }
    this->number_associations -= val;
}

int MapPoint::predict_image_scale(double distance) {
    float ratio = this->dmax / distance;
    int scale = ceil(log(ratio) / log(1.2));
    scale = (scale < 0) ? 0 : scale;
    scale = (scale >= 8) ? scale - 1 : scale;
    return scale;
}

void MapPoint::compute_view_direction() {
    if (this->data.size() == 0) return;
    Eigen::Vector3d normal = Eigen::Vector3d(0, 0, 0);
    Eigen::Vector3d current_normal;
    MapPointEntry *mp_entry;
    for (KeyFrame *kf : this->keyframes) {
        mp_entry = this->data[kf];
        current_normal = this->wcoord_3d - mp_entry->camera_center;
        current_normal.normalize();
        normal += current_normal;
    }
    this->view_direction = normal / (int)this->keyframes.size(); 
}

void MapPoint::compute_distance() {
    if (this->data.size() == 0) return;
    MapPointEntry *mp_entry = this->data[this->keyframes.front()];
    double distance = (this->wcoord_3d - mp_entry->camera_center).norm();
    this->dmax = distance * pow(1.2, octave);
    this->dmin = this->dmax / pow(1.2, 7);
    return;
}

void MapPoint::compute_distinctive_descriptor() {
    if (this->data.size() == 0) return;
    if (this->data.size() == 1) {
        this->orb_descriptor = this->data[this->keyframes.back()]->descriptor;
        return;
    }
    int N = this->data.size();
    std::vector<std::vector<int>> dist_mat(N, std::vector<int>(N));
    for (int i = 0; i < N - 1; i++) {
        dist_mat[i][i] = 0;
        for (int j = i + 1; j < N; j++) {
            float dist = ComputeHammingDistance(this->data[this->keyframes[i]]->descriptor, this->data[this->keyframes[j]]->descriptor);
            dist_mat[i][j] = dist;
            dist_mat[j][i] = dist; 
        }
    }
    int BestMedian = 10000;
    int BestIdx = 0;
    for(int i = 0; i < N; i++)
    {
        sort(dist_mat[i].begin(), dist_mat[i].end());
        int median = dist_mat[i][0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }
    this->orb_descriptor =  this->data[this->keyframes[BestIdx]]->descriptor;
}
