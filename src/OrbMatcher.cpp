#include "../include/OrbMatcher.h"


const int DES_DIST = 50;
int inline::OrbMatcher::ComputeHammingDistance(const cv::Mat &a, const cv::Mat &b)
{
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

void OrbMatcher::checkOrientation(KeyFrame *curr_kf, KeyFrame *prev_kf)
{
    std::vector<std::vector<MapPoint *>> histogram(30, std::vector<MapPoint *>());
    const float factor = 1.0f / 30;
    for (auto it = curr_kf->mp_correlations.begin(); it != curr_kf->mp_correlations.end(); it++)
    {
        MapPoint *mp = it->first;
        cv::KeyPoint current_kpu = it->second->get_undistorted_keypoint();
        if (prev_kf->mp_correlations.find(mp) == prev_kf->mp_correlations.end()) 
        {
            if (mp->keyframes.size() == 0) {
                std::cout << "NU ARE NICIUN KEYFRAME ASOCIAT\n";
            }
            std::cout << "printam keyframe-uri punct gresit\n";
            for (KeyFrame *kf : mp->keyframes) {
                std::cout << kf->current_idx << " acesta este un index al unui frame gasit\n";
            }
            std::cout << "E CIUDAT CA E NULL IN CHECK ORIENTATION\n";
            exit(1);
            continue;
        }
        cv::KeyPoint prev_kpu = prev_kf->mp_correlations[mp]->get_undistorted_keypoint();
        float rot = prev_kpu.angle - current_kpu.angle;
        if (rot < 0.0) rot += 360.0f;
        int bin = lround(rot * factor);
        if (bin == 30) bin = 0;
        histogram[bin].push_back(mp);
    }
    int max = 0;
    int index = 0;
    for (int i = 0; i < 30; i++) {
        if ((int)histogram[i].size() > max) {
            max = histogram[i].size();
            index = i;
        }
    }
    
    int threshold = 0.7 * max;
    std::vector<bool> keep(30, false);
    
    for (int i = 0; i < 30; i++) {
        if ((int)histogram[i].size() > threshold || 
            (i == ((index+1)%30) || i == ((index+29)%30))) {
            keep[i] = true;
        }
    }
    for (int i = 0; i < 30; i++) {
        if (keep[i]) continue;
        for (auto mp : histogram[i]) {
            Map::remove_map_point_from_keyframe(curr_kf, mp);
        }
    }
}

void OrbMatcher::match_consecutive_frames(KeyFrame *kf, KeyFrame *prev_kf, int window)
{
    Eigen::Vector3d kf_camera_center = kf->camera_center_world;
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector4d point_camera_coordinates;
    Eigen::Vector3d kf_camer_center_prev_coordinates = prev_kf->Tcw.rotationMatrix() * kf_camera_center + prev_kf->Tcw.translation();
    // aici era inainte 3.2 dar nu e corect este 0.08 -> stereo baseline in meters 
    const bool bForward  = kf_camer_center_prev_coordinates(2) >  0.08;
    const bool bBackward = -kf_camer_center_prev_coordinates(2) > 0.08;
    for (Feature f : prev_kf->features) {
    // for (auto it = prev_kf->mp_correlations.begin(); it != prev_kf->mp_correlations.end(); it++) {
    //     MapPoint *mp = it->first;
        MapPoint *mp = f.get_map_point();
        if (mp == nullptr || kf->check_map_point_outlier(mp)) continue;
        point_camera_coordinates = kf->mat_camera_world * mp->wcoord;
        if (point_camera_coordinates(2) <= 1e-6) continue;
        double d = point_camera_coordinates(2);
        double u = kf->K(0, 0) * point_camera_coordinates(0) / d + kf->K(0, 2);
        if (u < kf->minX || u > kf->maxX - 1) continue;
        double v = kf->K(1, 1) * point_camera_coordinates(1) / d + kf->K(1, 2);
        if (v < kf->minY || v > kf->maxY - 1) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        double distance = camera_to_map_view_ray.norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue; 

        int octave = prev_kf->mp_correlations[mp]->get_key_point().octave;
        int scale_of_search = window * kf->POW_OCTAVE[octave];
        std::vector<int> kps_idx;
        if (bForward) {
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, octave - 1, 8);
        } else if (bBackward) {
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, 0, octave);
        } else {
            kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, octave - 1, octave + 1);
        }
        if (kps_idx.size() == 0) continue;
        
        int best_dist = 256;
        int best_idx = -1;
        
        for (int idx : kps_idx)
        {
            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            if (kf->features[idx].stereo_depth > 1e-6)
            {
                double fake_rgbd = u - kf->K(0, 0) * 0.08 / d;
                float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
                if (er > scale_of_search) continue;
            }
            if (cur_hamm_dist < best_dist)
            {
                best_dist = cur_hamm_dist;
                best_idx = idx;
            }
        }
        if (best_dist > 100) continue;
        if (!Map::check_new_map_point_better(&kf->features[best_idx], mp)) continue;
        Map::add_map_point_to_keyframe(kf, &kf->features[best_idx], mp);
    }
    this->checkOrientation(kf, prev_kf);
}

void OrbMatcher::match_frame_map_points(KeyFrame *kf, std::unordered_set<MapPoint *>& map_points, int window_size)
{
    Eigen::Vector3d kf_camera_center = kf->camera_center_world;
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector4d point_camera_coordinates;
    
    for (MapPoint *mp : map_points) {
        if (kf->check_map_point_outlier(mp)) continue;
        point_camera_coordinates = kf->mat_camera_world * mp->wcoord;
        if (point_camera_coordinates(2) <= 1e-6) continue;
        double d = point_camera_coordinates(2);
        double u = kf->K(0, 0) * point_camera_coordinates(0) / d + kf->K(0, 2);
        if (u < kf->minX || u > kf->maxX - 1) continue;
        double v = kf->K(1, 1) * point_camera_coordinates(1) / d + kf->K(1, 2);
        if (v < kf->minY || v > kf->maxY - 1) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        double distance = camera_to_map_view_ray.norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;

        float radius = dot_product >= 0.998 ? 2.5 : 4;
        radius *= window_size;
        int predicted_scale = mp->predict_image_scale(distance);
        int scale_of_search = radius * kf->POW_OCTAVE[predicted_scale];

        int lowest_idx = -1;
        int lowest_dist = 256;
        int lowest_level = -1;

        int second_lowest_dist = 256;
        int second_lowest_level = -1;
        std::vector<int> kps_idx = kf->get_vector_keypoints_after_reprojection(u, v, scale_of_search, predicted_scale - 1, predicted_scale + 1);
        if (kps_idx.size() == 0) continue;
        for (int idx : kps_idx)
        {
            if (kf->features[idx].stereo_depth > 1e-6) {
                double fake_rgbd = u - kf->K(0, 0) * 0.08 / d;
                float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
                if (er > scale_of_search) continue;
            }

            int cur_hamm_dist = ComputeHammingDistance(mp->orb_descriptor, kf->features[idx].descriptor);
            
            if (cur_hamm_dist < lowest_dist)
            {
                second_lowest_dist = lowest_dist;
                second_lowest_level = lowest_level;
                lowest_idx = idx;
                lowest_dist = cur_hamm_dist;
                lowest_level = kf->features[idx].get_undistorted_keypoint().octave;
            }
            else if (cur_hamm_dist < second_lowest_dist)
            {
                second_lowest_dist = cur_hamm_dist;
                second_lowest_level = kf->features[idx].get_undistorted_keypoint().octave;
            }
        }
        if (lowest_dist > DES_DIST)  continue;
        if(lowest_level == second_lowest_level && lowest_dist > 0.7 * second_lowest_dist) continue;
        if (!Map::check_new_map_point_better(&kf->features[lowest_idx], mp)) continue;
        Map::add_map_point_to_keyframe(kf, &kf->features[lowest_idx], mp);
    }
}


void OrbMatcher::match_frame_reference_frame(KeyFrame *curr, KeyFrame *ref)
{
    curr->compute_bow_representation();
    DBoW2::FeatureVector::const_iterator f1it = ref->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f1end = ref->features_vec.end();
    DBoW2::FeatureVector::const_iterator f2it = curr->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f2end = curr->features_vec.end();
    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            for (int feature_idx : f1it->second)
            {
                MapPoint *mp_ref = ref->features[feature_idx].get_map_point();
                if (mp_ref == nullptr) continue;
                const cv::Mat &d1 = ref->orb_descriptors.row(feature_idx);
                
                int bestDist1 = 256;
                int bestIdx = -1;
                int bestDist2 = 256;
                
                for (int feature_curr : f2it->second)
                {
                    const cv::Mat &d2 = curr->orb_descriptors.row(feature_curr);
                    int dist = this->ComputeHammingDistance(d1, d2);
                    
                    if (dist < bestDist1)
                    {
                        bestDist2 = bestDist1;
                        bestDist1 = dist;
                        bestIdx = feature_curr;
                    }
                    else if (dist < bestDist2)
                    {
                        bestDist2 = dist;
                    }
                }
                if (bestDist1 > DES_DIST) continue;
                if (bestDist1 > 0.7 * bestDist2) continue;
                if (!Map::check_new_map_point_better(&curr->features[bestIdx], mp_ref)) continue;
                Map::add_map_point_to_keyframe(curr, &curr->features[bestIdx], mp_ref); 
            }
            f1it++;
            f2it++;
        }
        else if (f1it->first < f2it->first)
        {
            f1it = ref->features_vec.lower_bound(f2it->first);
        }
        else
        {
            f2it = curr->features_vec.lower_bound(f1it->first);
        }
    }
    this->checkOrientation(curr, ref);
}

std::vector<std::pair<int, int>> OrbMatcher::search_for_triangulation(KeyFrame *ref1, KeyFrame *ref2, Eigen::Matrix3d fundamental_matrix) {
    DBoW2::FeatureVector::const_iterator f1it = ref1->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f2it = ref2->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f1end = ref1->features_vec.end();
    DBoW2::FeatureVector::const_iterator f2end = ref2->features_vec.end();
    Eigen::Vector3d C2 = ref2->Tcw.rotationMatrix() * ref1->camera_center_world  + ref2->Tcw.translation();
    const float invz = 1.0f/C2(2);
    const float ex = ref2->K(0, 0) * C2(0) * invz + ref2->K(0, 2);
    const float ey = ref2->K(1, 1) * C2(1) * invz + ref2->K(1, 2);
    bool bStereo1 = false, bStereo2 = false;
    std::vector<std::pair<int, int>> vMatchedPairs;
    std::vector<bool> vbMatched2(ref2->features.size(),false);
    std::vector<int> vMatches12(ref1->features.size(),-1);
    int total_founds = 0;
    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t idx1 : f1it->second)
            {
                MapPoint* mp = ref1->features[idx1].get_map_point();
                if (mp != nullptr) continue;
                cv::KeyPoint kpu1 = ref1->features[idx1].get_undistorted_keypoint();
                const cv::Mat &d1 = ref1->orb_descriptors.row(idx1);
                int bestDist = 50;
                int bestIdx2 = -1;
                bStereo1 = ref1->features[idx1].stereo_depth > 1e-6;

                for(size_t idx2 : f2it->second)
                {
                    MapPoint* kf2_mp = ref2->features[idx2].get_map_point();
                    if (vbMatched2[idx2] || kf2_mp != nullptr) continue;
                    
                    const cv::Mat &d2 = ref2->orb_descriptors.row(idx2);
                    
                    const int dist = ComputeHammingDistance(d1,d2);
                    
                    if(dist>bestDist) continue;

                    const cv::KeyPoint &kpu2 = ref2->features[idx2].get_undistorted_keypoint();

                    bStereo2 = ref2->features[idx2].stereo_depth > 1e-6;
                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex - kpu2.pt.x;
                        const float distey = ey - kpu2.pt.y;
                        if(distex*distex+distey*distey < 100 * pow(1.2, kpu2.octave)) continue;
                    } 
                   
                    if(CheckDistEpipolarLine(kpu1, kpu2, fundamental_matrix))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                if(bestIdx2 < 0) continue;                    
                total_founds++;
                vMatches12[idx1]=bestIdx2;
                vbMatched2[bestIdx2] = true;
            }
            f1it++;
            f2it++;
        } else if(f1it->first < f2it->first)
        {
            f1it = ref1->features_vec.lower_bound(f2it->first);
        }
        else
        {
            f2it = ref2->features_vec.lower_bound(f1it->first);
        }
    }

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back({i, vMatches12[i]});
    }
    return vMatchedPairs;
} 


bool OrbMatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3d &F12)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    // Eigen::Vector3d line_parameters = F12 * Eigen::Vector3d(kp1.pt.y, kp1.pt.x, 1);
    Eigen::Vector3d line_parameters = Eigen::Vector3d(kp1.pt.x, kp1.pt.y, 1).transpose() * F12;
    const float a = line_parameters(0);
    const float b = line_parameters(1);
    const float c = line_parameters(2);

    const float den = a*a+b*b;
    if(den==0) return false;

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;
    const float dsqr = num * num / den;

    return dsqr < 3.84* pow(1.2, 2 * kp2.octave);
}


int OrbMatcher::Fuse(KeyFrame *pKF, KeyFrame *source_kf, const float th)
{
    Eigen::Matrix3d Rcw = pKF->Tcw.rotationMatrix();
    Eigen::Vector3d tcw = pKF->Tcw.translation();

    const float &fx = pKF->K(0, 0);
    const float &fy = pKF->K(1, 1);
    const float &cx = pKF->K(0, 2);
    const float &cy = pKF->K(1, 2);

    int nFused=0;
    std::unordered_set<MapPoint*> copy_map_points(source_kf->map_points.begin(), source_kf->map_points.end());
    int out = 0;
    for(MapPoint *source_mp : copy_map_points)
    {
        if(pKF->check_map_point_in_keyframe(source_mp)) continue;

        Eigen::Vector3d p3Dc = Rcw * source_mp->wcoord_3d + tcw;
        if(p3Dc(2) <= 1e-6) continue;

        const float invz = 1/p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        if (u < pKF->minX || u > pKF->maxX - 1) continue;
        if (v < pKF->minY || v > pKF->maxY - 1) continue;
        const float ur = u - fx * 0.08 * invz;

        Eigen::Vector3d camera_to_map_view_ray = (source_mp->wcoord_3d - pKF->camera_center_world);
        double distance = camera_to_map_view_ray.norm();
        if (distance < source_mp->dmin || distance > source_mp->dmax) continue;
        
        
        if(source_mp->view_direction.dot(camera_to_map_view_ray) < 0.5 * distance)
            continue;

        int nPredictedLevel = source_mp->predict_image_scale(distance);
        const float radius = th * pKF->POW_OCTAVE[nPredictedLevel];

        const std::vector<int>  vIndices = pKF->get_vector_keypoints_after_reprojection(u, v, radius, -1, 9); 

        if(vIndices.empty()) continue;

        // Match to the most similar keypoint in the radius
        int bestDist = 256;
        int bestIdx = -1;
        for(int kp_idx : vIndices)
        {
            const cv::KeyPoint &kpu = pKF->features[kp_idx].get_undistorted_keypoint();
            if(kpu.octave < nPredictedLevel-1 || kpu.octave > nPredictedLevel) continue;
            const float ex = u  - kpu.pt.x;
            const float ey = v  - kpu.pt.y;
            if (pKF->features[kp_idx].stereo_depth <= 1e-6) {
                const float e2 = ex*ex+ey*ey;
                if(e2 / pow(1.2, 2 * kpu.octave) > 5.99) continue;
            } 
            if (pKF->features[kp_idx].stereo_depth > 1e-6) {
                const float er = ur - pKF->features[kp_idx].stereo_depth;
                const float e2 = ex*ex+ey*ey+er*er;
                if(e2 / pow(1.2, 2 * kpu.octave) > 7.8) continue;
            }

            const int dist = ComputeHammingDistance(source_mp->orb_descriptor, pKF->features[kp_idx].descriptor);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = kp_idx;
            }
        }
        // If there is already a MapPoint replace otherwise add new measurement
        if (bestDist > 50) continue;
        out++;
        MapPoint* pkf_mp = pKF->features[bestIdx].get_map_point();
        if (pkf_mp == nullptr) {
            Feature *f = &pKF->features[bestIdx];
            bool map_to_kf = Map::add_map_point_to_keyframe(pKF, f, source_mp);
            bool kf_to_map = Map::add_keyframe_reference_to_map_point(source_mp, f, pKF);
            if (!map_to_kf ||  !kf_to_map) {
                std::cout << "NU S-A PUTUT ADAUGA MAP POINT TO KEYFRAME IN ORB MATCHER\n";
                continue;
            }
            nFused++;
            continue;
        }

        if(pkf_mp != nullptr && pkf_mp->keyframes.size() > source_mp->keyframes.size()) {
            Map::replace_map_point(source_mp, pkf_mp);
            nFused++;
            continue;
        }

        if (pkf_mp != nullptr && pkf_mp->keyframes.size() < source_mp->keyframes.size()) {
            Map::replace_map_point(pkf_mp, source_mp);
            nFused++;
            continue;
        }
    }
    return nFused;
}
