#include "../include/MapPoint.h"


int ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b) {
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


void inline::MapPoint::add_observation_map_point(KeyFrame *kf, cv::Mat descriptor, Eigen::Vector3d camera_center_kf) {
    this->keyframes.insert(kf);
    this->number_associations += 1;
    this->descriptor_vector.push_back(descriptor);
    int N = this->descriptor_vector.size();
    if (N == 1) {
        this->orb_descriptor = descriptor;

        return;
    }
    this->orb_descriptor = compute_distinctive_descriptor();
    this->view_direction = compute_view_direction(camera_center_kf);
}

MapPoint::MapPoint(KeyFrame *keyframe, cv::KeyPoint kp, Eigen::Vector3d camera_center, Eigen::Vector4d wcoord, 
        cv::Mat orb_descriptor) : wcoord(wcoord)
{
    this->wcoord_3d = Eigen::Vector3d(this->wcoord(0), this->wcoord(1), this->wcoord(2));
    this->add_observation_map_point(keyframe, orb_descriptor, camera_center);
    this->view_direction = this->compute_view_direction(camera_center);

    double distance = (wcoord_3d - camera_center).norm();
    this->view_direction.normalize();
    this->dmax = distance * pow(1.2, kp.octave);
    this->dmin = this->dmax / pow(1.2, 7);
}

void MapPoint::increase_how_many_times_seen() {
    this->number_times_seen += 1;
}

void MapPoint::increase_number_associations() {
    this->number_associations += 1;
}

void MapPoint::decrease_number_associations() {
    if (this->number_associations == 0) {
        std::cout << "NU POATE FI SCAZUT MAI TARE NUMARUL DE ASOCIERI DE 0\n";
    }
    this->number_associations -= 1;
}

int MapPoint::predict_image_scale(double distance) {
    float ratio = this->dmax / distance;
    int scale = ceil(log(ratio) / log(1.2));
    scale = (scale < 0) ? 0 : scale;
    scale = (scale >= 8) ? scale - 1 : scale;
    return scale;
}

Eigen::Vector3d MapPoint::compute_view_direction(Eigen::Vector3d camera_center) {
    if (this->keyframes.size() == 0 || this->keyframes.size() == 1) {
        Eigen::Vector3d normal = (wcoord_3d - camera_center);
        normal.normalize();
        return normal;
    }
    Eigen::Vector3d normal = this->wcoord_3d - camera_center;
    int N = this->keyframes.size();
    return this->view_direction * N / (N + 1)  + normal / (N + 1); 
}

cv::Mat MapPoint::compute_distinctive_descriptor() {
    int N = this->descriptor_vector.size();
    if (N == 1) return this->descriptor_vector[0];
    std::vector<std::vector<int>> dist_mat(N, std::vector<int>(N));
    for (int i = 0; i < N - 1; i++) {
        dist_mat[i][i] = 0;
        for (int j = i + 1; j < N; j++) {
            float dist = ComputeHammingDistance(descriptor_vector[i], descriptor_vector[j]);
            dist_mat[i][j] = dist;
            dist_mat[j][i] = dist; 
        }
    }
    int BestMedian = 10000;
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {
        sort(dist_mat[i].begin(), dist_mat[i].end());
        int median = dist_mat[i][0.5*(N-1)];

        if(median<BestMedian)
        {
            BestMedian = median;
            BestIdx = i;
        }
    }
    return this->descriptor_vector[BestIdx].clone();
}
