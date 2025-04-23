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

void OrbMatcher::checkOrientation(std::unordered_map<MapPoint *, Feature *> &correlation_current_frame,
                std::unordered_map<MapPoint *, Feature *> &correlation_prev_frame)
{
    std::vector<std::vector<MapPoint *>> histogram(30, std::vector<MapPoint *>());
    const float factor = 1.0f / 30;
    for (auto it = correlation_current_frame.begin(); it != correlation_current_frame.end(); it++)
    {
        MapPoint *mp = it->first;
        cv::KeyPoint current_kpu = it->second->get_undistorted_keypoint();
        if (correlation_prev_frame.find(mp) == correlation_prev_frame.end()) 
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
        cv::KeyPoint prev_kpu = correlation_prev_frame[mp]->get_undistorted_keypoint();
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
            correlation_current_frame.erase(mp);
        }
    }
}

void OrbMatcher::match_consecutive_frames(std::unordered_map<MapPoint*, Feature*>& matches, KeyFrame *kf, KeyFrame *prev_kf, int window)
{
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center_world();
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector3d point_camera_coordinates;
    Eigen::Vector3d kf_camer_center_prev_coordinates = prev_kf->Tcw.rotationMatrix() * kf_camera_center + prev_kf->Tcw.translation(); 
    const bool bForward  =   kf_camer_center_prev_coordinates(2) >  3.2;
    const bool bBackward = -kf_camer_center_prev_coordinates(2) > 3.2;
    for (auto it = prev_kf->mp_correlations.begin(); it != prev_kf->mp_correlations.end(); it++) {
        MapPoint *mp = it->first;
        if (kf->check_map_point_outlier(mp)) continue;
        point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < kf->minX || point_camera_coordinates(0) > kf->maxX - 1) continue;
        if (point_camera_coordinates(1) < kf->minY || point_camera_coordinates(1) > kf->maxY - 1) continue;
        if (point_camera_coordinates(2) < 1e-6) continue;

        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        double distance = camera_to_map_view_ray.norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue; 
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);
        
        int octave = prev_kf->mp_correlations[mp]->get_key_point().octave;
        int scale_of_search = window * pow(1.2, octave);
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
                double fake_rgbd = u - kf->K(0, 0) * 0.08 / point_camera_coordinates(2);
                float er = fabs(fake_rgbd - kf->features[idx].stereo_depth);
                if (er > scale_of_search) continue;
            }
            if (cur_hamm_dist < best_dist)
            {
                best_dist = cur_hamm_dist;
                best_idx = idx;
            }
        }
        if (best_dist > DES_DIST) continue;
        matches.insert({mp, &kf->features[best_idx]});
    }
    this->checkOrientation(matches, prev_kf->mp_correlations);
}

void OrbMatcher::match_frame_map_points(std::unordered_map<MapPoint*, Feature*>& matches, KeyFrame *kf, std::unordered_set<MapPoint *>& map_points, int window_size)
{
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center_world();
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector3d point_camera_coordinates;
    
    for (MapPoint *mp : map_points) {
        if (kf->check_map_point_outlier(mp)) continue;
        point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < kf->minX || point_camera_coordinates(0) > kf->maxX - 1) continue;
        if (point_camera_coordinates(1) < kf->minY || point_camera_coordinates(1) > kf->maxY - 1) continue;
        if (point_camera_coordinates(2) < 1e-6) continue;
        
        camera_to_map_view_ray = (mp->wcoord_3d - kf_camera_center);
        double distance = camera_to_map_view_ray.norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        camera_to_map_view_ray.normalize();
        double dot_product = camera_to_map_view_ray.dot(mp->view_direction);
        if (dot_product < 0.5) continue;
        mp->increase_how_many_times_seen(); 

        float radius = dot_product >= 0.998 ? 2.5 : 4;
        radius *= window_size;
        int predicted_scale = mp->predict_image_scale(distance);
        int scale_of_search = radius * pow(1.2, predicted_scale);        
        double u = point_camera_coordinates(0);
        double v = point_camera_coordinates(1);

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
                double fake_rgbd = u - kf->K(0, 0) * 0.08 / point_camera_coordinates(2);
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
        if(lowest_level == second_lowest_level && lowest_dist > 0.8 * second_lowest_dist) continue;
        matches.insert({mp, &kf->features[lowest_idx]});
    }
}


void OrbMatcher::match_frame_reference_frame(std::unordered_map<MapPoint*, Feature*>& matches, KeyFrame *curr, KeyFrame *ref)
{
    curr->compute_bow_representation();
    DBoW2::FeatureVector::const_iterator f1it = ref->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f1end = ref->features_vec.end();
    DBoW2::FeatureVector::const_iterator f2it = curr->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f2end = curr->features_vec.end();
    // std::cout << curr->map_points.size() << " atatea map points asociate la inceput cu curr\n";
    // std::cout << ref->map_points.size() << " atatea map points asociate la inceput cu ref\n";
    // std::cout << ref->current_idx << "index reference frame current\n"; 
    // std::cout << ref->map_points.size() << " " << ref->mp_correlations.size() << " " << " info about reference frame\n";
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
                matches.insert({mp_ref, &curr->features[bestIdx]});
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
    // std::cout << curr->current_idx << " " << ref->current_idx << " indicii in track local frame\n";
    this->checkOrientation(matches, ref->mp_correlations);
}

std::vector<std::pair<int, int>> OrbMatcher::search_for_triangulation(KeyFrame *ref1, KeyFrame *ref2, Eigen::Matrix3d fundamental_matrix) {
    DBoW2::FeatureVector::const_iterator f1it = ref1->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f2it = ref2->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f1end = ref1->features_vec.end();
    DBoW2::FeatureVector::const_iterator f2end = ref2->features_vec.end();
    Eigen::Vector3d C2 = ref2->Tcw.rotationMatrix() * ref1->compute_camera_center_world() + ref2->Tcw.translation();
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
                    if (kf2_mp != nullptr) continue;
                    if(vbMatched2[idx2]) continue;
                    
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

// NU ESTE CLAR EPIPOLAR CONSTRAINS PART
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
    const float &bf = 3.2;

    Eigen::Vector3d Ow = pKF->compute_camera_center_world();

    int nFused=0;
    bool was_deletion_sucessfull, was_addition_succesful; 
    std::unordered_set<MapPoint*> copy_map_points(source_kf->map_points.begin(), source_kf->map_points.end());
    for(MapPoint *source_mp : copy_map_points)
    {
        if(source_mp->keyframes.find(pKF) != source_mp->keyframes.end()) continue;

        Eigen::Vector3d p3Dc = Rcw * source_mp->wcoord_3d + tcw;
        if(p3Dc(2)<0.0f) continue;

        const float invz = 1/p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        if (u < pKF->minX || u > pKF->maxX - 1) continue;
        if (v < pKF->minY || v > pKF->maxY - 1) continue;
        const float ur = u-bf*invz;

        Eigen::Vector3d camera_to_map_view_ray = (source_mp->wcoord_3d - Ow);
        double distance = camera_to_map_view_ray.norm();
        if (distance < source_mp->dmin || distance > source_mp->dmax) continue;
        
        
        if(source_mp->view_direction.dot(camera_to_map_view_ray) < 0.5 * distance)
            continue;

        int nPredictedLevel = source_mp->predict_image_scale(distance);
        const float radius = th * pow(1.2, nPredictedLevel);

        const std::vector<int>  vIndices = pKF->get_vector_keypoints_after_reprojection(u, v, radius, -1, 9); 

        if(vIndices.empty()) continue;

        // Match to the most similar keypoint in the radius
        int bestDist = 1000;
        int bestIdx = -1;
        for(int kp_idx : vIndices)
        {
            const cv::KeyPoint &kpu = pKF->features[kp_idx].get_undistorted_keypoint();

            const int &kpLevel= kpu.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel) continue;

            if(pKF->features[kp_idx].stereo_depth >= 0)
            {
                // Check reprojection error in stereo
                const float &kpx = kpu.pt.x;
                const float &kpy = kpu.pt.y;
                const float &kpr = pKF->features[kp_idx].stereo_depth;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;
                if(e2 * pow(1.2, 2 * kpLevel) > 7.8) continue;
            }
            else
            {
                const float &kpx = kpu.pt.x;
                const float &kpy = kpu.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;
                if(e2 * pow(1.2, 2 * kpLevel) > 5.99) continue;
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
        MapPoint* pkf_mp = pKF->features[bestIdx].get_map_point();
        if (pkf_mp == nullptr) {
            was_addition_succesful = Map::add_map_point_to_keyframe(pKF, &pKF->features[bestIdx], source_mp);
            if (!was_addition_succesful) {
                std::cout << "NU S-A PUTUT ADAUGA MAP POINT DESI ERA NULL MAP POINT LA FEATURE\n";
                continue;
            } 
            was_addition_succesful = Map::add_keyframe_reference_to_map_point(source_mp, pKF);
            if (!was_addition_succesful) {
                std::cout << "NU S-A PUTUT ADAUGA MAP POINT TO KEYFRAME IN ORB MATCHER\n";
                continue;
            }
            nFused++;
            continue;
        }
        if(pkf_mp != nullptr && pkf_mp->keyframes.size() > source_mp->keyframes.size()) {
            // retire source_mp,
            // replace it pkf_mp; 
            // nu are sens linia asta
            if (source_kf->check_map_point_in_keyframe(pkf_mp)) continue; 
            Feature *f = source_kf->mp_correlations[source_mp];
            was_deletion_sucessfull = Map::remove_map_point_from_keyframe(source_kf, source_mp);
            if (!was_deletion_sucessfull) {
                std::cout << "NU S-A PUTUT STERGEREA NU A FUNCTION IN FUSE\n";
                continue;
            }
            was_addition_succesful = Map::add_map_point_to_keyframe(source_kf, f, pkf_mp);
            if (!was_addition_succesful) {
                std::cout << "NU S-A PUTUT SA ADAUGE PUNCTUL IN FUSE\n";
                continue;
            }
            was_addition_succesful = Map::add_keyframe_reference_to_map_point(pkf_mp, source_kf);
            if (!was_addition_succesful) {
                std::cout << "NU S-A PUTUT SA ADAUGE PUNCTUL IN FUSE\n";
                continue;
            }
            nFused++;
            continue;
        }
        if (pkf_mp != nullptr && pkf_mp->keyframes.size() < source_mp->keyframes.size()) {
            // retire pkf_mp;
            // replace it with source_mp;
            if (pKF->check_map_point_in_keyframe(source_mp)) continue;
            Feature *f = pKF->mp_correlations[pkf_mp];
            was_deletion_sucessfull = Map::remove_map_point_from_keyframe(pKF, pkf_mp);
            if (!was_deletion_sucessfull) {
                std::cout << "NU S-A PUTUT REALIZA STERGEREA\n";
                continue;
            }
            was_addition_succesful = Map::add_map_point_to_keyframe(pKF, f, source_mp);
            if (!was_addition_succesful) {
                std::cout << "NU S-A PUTUT SA ADAUGE PUNCTUL IN FUSE SECOND\n";
                continue;
            }
            was_addition_succesful = Map::add_keyframe_reference_to_map_point(source_mp, pKF);
            if (!was_addition_succesful) {
                std::cout << "NU S-A PUTUT SA ADAUGE PUNCTUL IN FUSE SECOND\n";
                continue;
            }
            nFused++;
            continue;
        }
    }
    return nFused;
}
