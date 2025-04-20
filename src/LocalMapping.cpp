#include "../include/LocalMapping.h"


void LocalMapping::local_map(KeyFrame *kf) {
    std::cout << "start local map\n";
    mapp->add_new_keyframe(kf);
    this->map_points_culling(kf);
    int map_points_computed = this->compute_map_points(kf);
    std::cout << map_points_computed << " MAP POINT-uri noi create in acest frame\n";
    if (map_points_computed == 0) std::cout << "NICIUN MAP POINT NU A FOST CALCULAT\n";
    this->search_in_neighbours(kf);
    this->update_local_map(kf);
    std::cout << "end local map\n";
}


void LocalMapping::map_points_culling(KeyFrame *kf) {
    std::vector<MapPoint*> to_del;
    // se strica ordinea din unordered set si se pierd elemente daca fac stergerea direct
    for (MapPoint *mp : kf->reference_kf->map_points) {
        if (mp->map_point_should_be_deleted()) to_del.push_back(mp);
    }
    for (MapPoint *mp : to_del) this->delete_map_point(mp);
}



bool is_coordinate_valid_for_keyframe(KeyFrame *kf, Feature *f, Eigen::Vector4d coordinates,  bool isStereo) {
    Eigen::Vector3d projected_on_kf = kf->Tcw.matrix3x4() * coordinates;
    const float z1 = projected_on_kf(2);
    if (z1 <= 1e-6) return false;
    const float invz1 = 1.0 / z1;
    const float &sigmaSquare1 = pow(1.2, 2 * f->get_undistorted_keypoint().octave);
    float u1 = kf->K(0, 0) * projected_on_kf(0) * invz1 + kf->K(0, 2);
    float v1 = kf->K(1, 1) * projected_on_kf(1) * invz1 + kf->K(1, 2);
    float errX1 = u1 - f->get_undistorted_keypoint().pt.x;
    float errY1 = v1 - f->get_undistorted_keypoint().pt.y;
    
    if (isStereo) {
        float u1_r = u1 - kf->K(0, 0) * 0.08 * invz1;
        float errX1_r = u1_r - f->stereo_depth;
        return (errX1 * errX1 + errY1 * errY1+ errX1_r * errX1_r) < 7.8 * sigmaSquare1; 
    }
    return (errX1 * errX1 + errY1 * errY1) < 5.991 * sigmaSquare1;
}


int LocalMapping::compute_map_points(KeyFrame *kf)
{ 
    double fx1 = kf->K(0, 0);
    double fy1 = kf->K(1, 1);
    double cx1 = kf->K(0, 2);
    double cy1 = kf->K(1, 2);
    double invfy1 = 1 / kf->K(1, 2);
    double invfx1 = 1 / kf->K(0, 2);
    std::unordered_set<KeyFrame*> keyframes = mapp->get_local_keyframes(kf);

    bool isStereo1 = false;
    bool isStereo2 = false;
    int nr_points_added = 0;
    for (KeyFrame* neighbour_kf : keyframes) {
        double fx2 = neighbour_kf->K(0, 0);
        double fy2 = neighbour_kf->K(1, 1);
        double cx2 = neighbour_kf->K(0, 2);
        double cy2 = neighbour_kf->K(1, 2);
        double invfx2 = 1 / neighbour_kf->K(0, 2);
        double invfy2 = 1 / neighbour_kf->K(1, 2);
        // TODO: check wheter the keyframe is far enough from the map point
        Eigen::Matrix3d fundamental_mat =  this->compute_fundamental_matrix(kf, neighbour_kf);
        std::vector<std::pair<int, int>> vMatchedIndices = OrbMatcher::search_for_triangulation(kf, neighbour_kf, fundamental_mat);
        //         std::vector<cv::DMatch> dmatches;
        // for (std::pair<int, int> correspondence : vMatchedIndices) {
        //         cv::DMatch dmatch;
        //         dmatch.queryIdx = correspondence.first;
        //         dmatch.trainIdx = correspondence.second;
        //         dmatch.distance = OrbMatcher::ComputeHammingDistance(kf->features[dmatch.queryIdx].descriptor, neighbour_kf->features[dmatch.trainIdx].descriptor); 
        //         dmatches.push_back(dmatch);
        //     }
        //     cv::Mat img_matches;
        //     std::cout << dmatches.size() << " atatea match-uri sunt posibile\n";
        //     cv::drawMatches(kf->frame, kf->get_all_keypoints(), neighbour_kf->frame, neighbour_kf->get_all_keypoints(), dmatches, img_matches);
        //     cv::imshow("Feature Matches", img_matches);
        //     cv::waitKey(0);
        for (std::pair<int, int> correspondence : vMatchedIndices) {
            Feature *f1 = &kf->features[correspondence.first];
            Feature *f2 = &neighbour_kf->features[correspondence.second];
            isStereo1 = f1->stereo_depth > 1e-6;
            isStereo2 = f2->stereo_depth > 1e-6;

            double new_x1 = (f1->get_undistorted_keypoint().pt.x - kf->K(0, 2)) / kf->K(0, 0);
            double new_y1 = (f1->get_undistorted_keypoint().pt.y - kf->K(1, 2)) / kf->K(1, 1);
            Eigen::Vector3d normalised_first_point(new_x1, new_y1, 1.0); 


            double new_x2 = (f2->get_undistorted_keypoint().pt.x - neighbour_kf->K(0, 2)) / neighbour_kf->K(0, 0);
            double new_y2 = (f2->get_undistorted_keypoint().pt.y - neighbour_kf->K(1, 2)) / neighbour_kf->K(1, 1);
            Eigen::Vector3d normalised_second_point(new_x2, new_y2, 1.0);

            Eigen::Vector3d ray1 = kf->Tcw.rotationMatrix().transpose() *  normalised_first_point;
            Eigen::Vector3d ray2 = neighbour_kf->Tcw.rotationMatrix().transpose() * normalised_second_point;

            const float cosParallaxRays = ray1.dot(ray2) / (ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(isStereo1)
                cosParallaxStereo1 = cos(2*atan2(0.04, f1->depth ));
            else if(isStereo2)
                cosParallaxStereo2 = cos(2*atan2(0.04, f2->depth));

            cosParallaxStereo = std::min(cosParallaxStereo1,cosParallaxStereo2);

            Eigen::Vector4d coordinates;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (isStereo1 || isStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                Eigen::Matrix4d A;
                A.row(0) = normalised_first_point(0) *kf->Tcw.matrix().row(2)-kf->Tcw.matrix().row(0);
                A.row(1) = normalised_first_point(1) *kf->Tcw.matrix().row(2)-kf->Tcw.matrix().row(1);
                A.row(2) = normalised_second_point(0) * neighbour_kf->Tcw.matrix().row(2) - neighbour_kf->Tcw.matrix().row(0);
                A.row(3) = normalised_second_point(1) * neighbour_kf->Tcw.matrix().row(2) - neighbour_kf->Tcw.matrix().row(1);

                cv::Mat w,u, vt, A_mat;
                cv::eigen2cv(A, A_mat);
                cv::SVD::compute(A_mat,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
                cv::Mat x3D = vt.row(3).t();

                if(x3D.at<double>(3) == 0)
                continue;
                
                // Euclidean coordinates
                x3D = x3D.rowRange(0,4)/x3D.at<double>(3);
                cv::cv2eigen(x3D, coordinates);
            }
            else if(isStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                coordinates = kf->fromImageToWorld(f1->idx);
            }
            else if(isStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                coordinates = neighbour_kf->fromImageToWorld(f2->idx);
            }
            else continue; 
            
            if (!is_coordinate_valid_for_keyframe(kf, f1, coordinates, isStereo1)) continue;
            if (!is_coordinate_valid_for_keyframe(neighbour_kf, f2, coordinates, isStereo2)) continue;
                
            //Check scale consistency
            Eigen::Vector3d coordinates_3d(coordinates(0), coordinates(1), coordinates(2)); 
            Eigen::Vector3d normal1 = coordinates_3d - kf->compute_camera_center_world();
            float dist1 = normal1.norm();

            Eigen::Vector3d normal2 = coordinates_3d - neighbour_kf->compute_camera_center_world();
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = pow(1.2 , f1->get_undistorted_keypoint().octave - f2->get_undistorted_keypoint().octave);

            if(ratioDist* 1.8 < ratioOctave || ratioDist>ratioOctave * 1.8)
                continue;

            MapPoint* pMP = new MapPoint(kf, f1->get_undistorted_keypoint(), kf->compute_camera_center_world(), coordinates,
                     kf->orb_descriptors.row(correspondence.first));
               
            // TODO, de adaugat si observatia in frame-ul al doilea 
            kf->add_map_point(pMP, &kf->features[correspondence.first], pMP->orb_descriptor);
            neighbour_kf->add_map_point(pMP, &neighbour_kf->features[correspondence.second], pMP->orb_descriptor);
            pMP->add_observation_map_point(neighbour_kf, neighbour_kf->features[correspondence.second].descriptor, kf->compute_camera_center_world());
            nr_points_added++;
            // TODOOOOOO
            // (1) UPDATE_DEPTH
        }
    }
    return nr_points_added;
}

void LocalMapping::search_in_neighbours(KeyFrame *kf) {
    std::unordered_set<KeyFrame*> vpNeighKFs = mapp->get_local_keyframes(kf);
    for(KeyFrame* pKFi :  vpNeighKFs)
    {
        std::unordered_set<KeyFrame*> vpSecondNeighKFs = mapp->get_local_keyframes(pKFi);
        vpNeighKFs.insert(vpSecondNeighKFs.begin(), vpSecondNeighKFs.end());
    }

    // Search matches by projection from current KF in target KFs
    vpNeighKFs.erase(kf);
    for(KeyFrame *neighbour_kf : vpNeighKFs)
    {
        OrbMatcher::Fuse(neighbour_kf, kf, 3);
    }
    // Search matches by projection from target KFs in current KF
    std::vector<MapPoint*> vpFuseCandidates;
    for(KeyFrame *neighbour_kf : vpNeighKFs)
    {
        OrbMatcher::Fuse(kf, neighbour_kf, 3);
    }
    
    // Update connections in covisibility graph
    // mpCurrentKeyFrame->UpdateConnections();
}


Eigen::Matrix3d LocalMapping::compute_fundamental_matrix(KeyFrame *curr_kf, KeyFrame *neighbour_kf) {
    Eigen::Matrix3d R1w = curr_kf->Tcw.rotationMatrix();
    Eigen::Vector3d t1w = curr_kf->Tcw.translation();
    Eigen::Matrix3d R2w = neighbour_kf->Tcw.rotationMatrix();
    Eigen::Vector3d t2w = neighbour_kf->Tcw.translation();

    Eigen::Matrix3d R12 = R1w*R2w.transpose();
    Eigen::Vector3d t12 = -R1w* R2w.transpose() * t2w + t1w;
 
    Eigen::Matrix3d t12x;
    t12x <<     0,   -t12.z(),  t12.y(),
          t12.z(),      0,  -t12.x(),
         -t12.y(),  t12.x(),     0;
    return curr_kf->K.transpose().inverse() * t12x* R12 * neighbour_kf->K.inverse();
}


void LocalMapping::delete_map_point(MapPoint *mp) {
    for (KeyFrame *kf : mp->keyframes) { 
        kf->remove_map_point(mp);
    }
    if (mapp->local_map.find(mp) != mapp->local_map.end()) mapp->local_map.erase(mp);
    // delete mp; dintr-un motiv sau altul eroare in momentul in care incerc sa sterg definitiv un map point - bug in cate map point-uri raman si cate sunt sterse garantat
}

void LocalMapping::update_local_map(KeyFrame *reference_kf)
{
    std::unordered_set<MapPoint *> out = reference_kf->map_points;
    if (this->mapp->graph.find(reference_kf) == this->mapp->graph.end()) {
        std::cout << "REFERENCE FRAME NU A FOST ADAUGAT IN GRAPH DELOC\n";
        return;
    }
    for (std::pair<KeyFrame *, int> graph_edge : this->mapp->graph[reference_kf])
    {
        KeyFrame *curr_kf = graph_edge.first;
        if (curr_kf == nullptr)  {
            std::cout << " nu e bine ca e null\n";
            continue;
        }
        out.insert(curr_kf->map_points.begin(), curr_kf->map_points.end());
    }
}