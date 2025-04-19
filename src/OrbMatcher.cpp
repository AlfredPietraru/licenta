#include "../include/OrbMatcher.h"


const int DES_DIST = 60;
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

std::unordered_map<MapPoint *, Feature *> OrbMatcher::checkOrientation(std::unordered_map<MapPoint *, Feature *> &correlation_current_frame,
                std::unordered_map<MapPoint *, Feature *> &correlation_prev_frame)
{
    std::vector<std::vector<MapPoint *>> histogram(30, std::vector<MapPoint *>());
    const float factor = 1.0f / 30;
    for (auto it = correlation_current_frame.begin(); it != correlation_current_frame.end(); it++)
    {
        MapPoint *mp = it->first;
        cv::KeyPoint current_kpu = it->second->get_undistorted_keypoint();
        if (correlation_prev_frame.find(mp) == nullptr) 
        {
            std::cout << "E CIUDAT CA E NULL IN CHECK ORIENTATION\n";
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
        if (histogram[i].size() > max) {
            max = histogram[i].size();
            index = i;
        }
    }
    
    int threshold = 0.7 * max;
    std::vector<bool> keep(30, false);
    
    for (int i = 0; i < 30; i++) {
        if (histogram[i].size() > threshold || 
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
    return correlation_current_frame;
}

void OrbMatcher::match_consecutive_frames(std::unordered_map<MapPoint*, Feature*>& matches, KeyFrame *kf, KeyFrame *prev_kf, int window)
{
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector3d point_camera_coordinates;
    Eigen::Vector3d kf_camer_center_prev_coordinates = prev_kf->Tiw.rotationMatrix() * kf_camera_center + prev_kf->Tiw.translation(); 
    const bool bForward  =   kf_camer_center_prev_coordinates(2) >  3.2;
    const bool bBackward = -kf_camer_center_prev_coordinates(2) > 3.2;
    for (auto it = prev_kf->mp_correlations.begin(); it != prev_kf->mp_correlations.end(); it++) {
        MapPoint *mp = it->first;
        if (kf->check_map_point_outlier(mp)) continue;
        point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < 0 || point_camera_coordinates(0) > kf->frame.cols - 1) continue;
        if (point_camera_coordinates(1) < 0 || point_camera_coordinates(1) > kf->frame.rows - 1) continue;
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
    Eigen::Vector3d kf_camera_center = kf->compute_camera_center();
    Eigen::Vector3d camera_to_map_view_ray;
    Eigen::Vector3d point_camera_coordinates;
    
    for (MapPoint *mp : map_points) {
        if (kf->check_map_point_outlier(mp)) continue;
        point_camera_coordinates = kf->fromWorldToImage(mp->wcoord);
        if (point_camera_coordinates(0) < 0 || point_camera_coordinates(0) > kf->frame.cols - 1) continue;
        if (point_camera_coordinates(1) < 0 || point_camera_coordinates(1) > kf->frame.rows - 1) continue;
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

        int lowest_dist = 256;
        int lowest_idx = -1;
        int lowest_level = -1;
        int second_lowest_dist = 256;
        int second_lowest_level = -1;
        int second_lowest_idx = -1;
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
                second_lowest_idx = lowest_idx;
                second_lowest_level = lowest_level;
                lowest_idx = idx;
                lowest_dist = cur_hamm_dist;
                lowest_level = kf->features[idx].get_undistorted_keypoint().octave;
            }
            else if (cur_hamm_dist < second_lowest_dist)
            {
                second_lowest_dist = cur_hamm_dist;
                second_lowest_idx = idx;
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
    
    while (f1it != f1end && f2it != f2end)
    {
        if (f1it->first == f2it->first)
        {
            std::vector<unsigned int> indices_ref = f1it->second;
            std::vector<unsigned int> indicies_curr = f2it->second;

            for (size_t iref = 0; iref < indices_ref.size();  iref++)
            {
                const size_t feature_idx = f1it->second[iref];
                MapPoint *mp_ref = ref->features[feature_idx].get_map_point();
                if (mp_ref == nullptr) continue;
                const cv::Mat &d1 = ref->orb_descriptors.row(feature_idx);
                
                int bestDist1 = 256;
                int bestIdx = -1;
                int bestDist2 = 256;
                
                for (size_t icurr = 0; icurr < indicies_curr.size(); icurr++)
                {
                    const size_t feature_curr = f2it->second[icurr];
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
    this->checkOrientation(matches, ref->mp_correlations);
}

std::vector<std::pair<int, int>> OrbMatcher::search_for_triangulation(KeyFrame *ref1, KeyFrame *ref2, Eigen::Matrix3d fundamental_matrix) {
    DBoW2::FeatureVector::const_iterator f1it = ref1->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f2it = ref2->features_vec.begin();
    DBoW2::FeatureVector::const_iterator f1end = ref1->features_vec.end();
    DBoW2::FeatureVector::const_iterator f2end = ref2->features_vec.end();
    Eigen::Vector3d Cw = ref1->compute_camera_center();
    Eigen::Matrix3d R2w = ref2->Tiw.rotationMatrix();
    Eigen::Vector3d t2w = ref2->Tiw.translation();
    Eigen::Vector3d C2 = R2w*Cw+t2w;
    const float invz = 1.0f/C2(2);
    const float ex = ref2->K(0, 0) * C2(0) * invz + ref2->K(0, 2);
    const float ey = ref2->K(1, 1) * C2(1) * invz + ref2->K(1, 2);
    bool bStereo1 = false, bStereo2 = false;
    std::vector<std::pair<int, int>> vMatchedPairs;
    std::vector<bool> vbMatched2(ref2->features.size(),false);
    std::vector<int> vMatches12(ref1->features.size(),-1);
    std::cout << "inainte sa inceapa triangularea\n";
    int total_founds = 0;
    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                MapPoint* mp = ref1->features[idx1].get_map_point();
                if (mp != nullptr) continue;
                cv::KeyPoint kpu1 = ref1->features[idx1].get_undistorted_keypoint();
                const cv::Mat &d1 = ref1->orb_descriptors.row(idx1);
                int bestDist = 50;
                int bestIdx2 = -1;
                bStereo1 = ref1->features[idx1].stereo_depth > 1e-6;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    MapPoint* kf2_mp = ref2->features[idx2].get_map_point();
                    
                    if(vbMatched2[idx2] || kf2_mp != nullptr)
                        continue;
                    
                    const cv::Mat &d2 = ref2->orb_descriptors.row(idx2);
                    
                    const int dist = ComputeHammingDistance(d1,d2);
                    
                    if(dist>bestDist) continue;

                    const cv::KeyPoint &kpu2 = ref2->features[idx2].get_undistorted_keypoint();

                    bStereo2 = ref2->features[idx2].stereo_depth > 1e-6;
                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kpu2.pt.x;
                        const float distey = ey-kpu2.pt.y;
                        if(distex*distex+distey*distey < 50 * pow(1.2, kpu2.octave)) continue;
                    } 
                    bestIdx2 = idx2;
                    bestDist = dist;
                   
                    if(CheckDistEpipolarLine(kpu1,kpu2, fundamental_matrix, ref2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                if(bestIdx2 < 0) continue;                    
                total_founds++;
                vMatches12[idx1]=bestIdx2;
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

bool OrbMatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const Eigen::Matrix3d &F12, const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12(0,0)+kp1.pt.y*F12(1,0)+F12(2,0);
    const float b = kp1.pt.x*F12(0,1)+kp1.pt.y*F12(1,1)+F12(2,1);
    const float c = kp1.pt.x*F12(0,2)+kp1.pt.y*F12(1,2)+F12(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr< 3.84* pow(1.2, kp2.octave);
}


int OrbMatcher::Fuse(KeyFrame *pKF, KeyFrame *source_kf, const float th)
{
    Eigen::Matrix3d Rcw = pKF->Tiw.rotationMatrix();
    Eigen::Vector3d tcw = pKF->Tiw.translation();

    const float &fx = pKF->K(0, 0);
    const float &fy = pKF->K(1, 1);
    const float &cx = pKF->K(0, 2);
    const float &cy = pKF->K(1, 2);
    const float &bf = 3.2;

    Eigen::Vector3d Ow = pKF->compute_camera_center();

    int nFused=0;

    const int nMPs = source_kf->map_points.size();

    for(MapPoint *mp : source_kf->map_points)
    {
        if(!mp) continue;
        if(mp->keyframes.find(pKF) != mp->keyframes.end()) continue;

        Eigen::Vector3d p3Dc = Rcw * mp->wcoord_3d + tcw;
        if(p3Dc(2)<0.0f) continue;

        const float invz = 1/p3Dc(2);
        const float x = p3Dc(0) * invz;
        const float y = p3Dc(1) * invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        if (u < 0 || u > pKF->frame.cols - 1) continue;
        if (v < 0 || v > pKF->frame.rows - 1) continue;
        const float ur = u-bf*invz;

        Eigen::Vector3d camera_to_map_view_ray = (mp->wcoord_3d - Ow);
        double distance = camera_to_map_view_ray.norm();
        if (distance < mp->dmin || distance > mp->dmax) continue;
        
        
        if(mp->view_direction.dot(camera_to_map_view_ray) < 0.5 * distance)
            continue;

        int nPredictedLevel = mp->predict_image_scale(distance);
        const float radius = th * pow(1.2, nPredictedLevel);

        const std::vector<int>  vIndices = pKF->get_vector_keypoints_after_reprojection(u, v, radius, -1, 9); 

        if(vIndices.empty()) continue;

        // Match to the most similar keypoint in the radius
        int bestDist = 256;
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
            const int dist = ComputeHammingDistance(mp->orb_descriptor, pKF->features[kp_idx].descriptor);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = kp_idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=50)
        {
            MapPoint* pMPinKF = pKF->features[bestIdx].get_map_point();
            if (pMPinKF == nullptr) {
                mp->add_observation_map_point(pKF, pKF->features[bestIdx].descriptor, Ow);
                pKF->add_map_point(mp, &pKF->features[bestIdx], mp->orb_descriptor);
                nFused++;
                continue;
            }
            if(pMPinKF != nullptr)
            {
                if(pMPinKF->keyframes.size() > mp->keyframes.size()) {
                    if (source_kf->mp_correlations.find(mp) == source_kf->mp_correlations.end()) {
                        std::cout << "NU VEDE PUNCTUL IN LOCAL MAPPING\n";
                        continue;
                    } 
                    Feature *f = source_kf->mp_correlations[mp];
                    source_kf->remove_map_point(mp);
                    source_kf->add_map_point(pMPinKF, f, mp->orb_descriptor);
                    continue;
                }
                if (pMPinKF->keyframes.size() < mp->keyframes.size()) {
                    if (pKF->mp_correlations.find(mp) == pKF->mp_correlations.end()) {
                        std::cout << "NU VEDE PUNCTUL IN LOCAL MAPPING\n";
                        continue;
                    } 
                    Feature *f = pKF->mp_correlations[mp];
                    pKF->remove_map_point(mp);
                    pKF->add_map_point(pMPinKF, f, mp->orb_descriptor);
                    continue;
                }
            }
            nFused++;
        }
    }

    return nFused;
}
