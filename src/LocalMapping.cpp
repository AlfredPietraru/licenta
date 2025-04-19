#include "../include/LocalMapping.h"


void LocalMapping::local_map(KeyFrame *kf) {
    mapp->add_new_keyframe(kf);
    this->map_points_culling(kf);
    this->compute_map_points(kf);
    this->search_in_neighbours(kf);
    this->update_local_map(kf);
}


void LocalMapping::map_points_culling(KeyFrame *kf) {
    std::vector<MapPoint*> to_del;
    // se strica ordinea din unordered set si se pierd elemente daca fac stergerea direct
    for (MapPoint *mp : kf->reference_kf->map_points) {
        if (mp->map_point_should_be_deleted()) to_del.push_back(mp);
    }
    for (MapPoint *mp : to_del) this->delete_map_point(mp);
}


void LocalMapping::compute_map_points(KeyFrame *kf)
{ 
    double fx1 = kf->K(0, 0);
    double fy1 = kf->K(1, 1);
    double cx1 = kf->K(0, 2);
    double cy1 = kf->K(1, 2);
    double invfy1 = 1 / kf->K(1, 2);
    double invfx1 = 1 / kf->K(0, 2);
    cv::Mat Rwc1;
    cv::eigen2cv(kf->Tiw.rotationMatrix(), Rwc1);


    std::unordered_set<KeyFrame*> keyframes = mapp->get_local_keyframes(kf);

    bool isStereo1 = false;
    bool isStereo2 = false;
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
        for (std::pair<int, int> correspondence : vMatchedIndices) {
            const cv::KeyPoint kpu1 = kf->features[correspondence.first].get_undistorted_keypoint();
            const cv::KeyPoint kpu2 = neighbour_kf->features[correspondence.second].get_undistorted_keypoint();
            isStereo1 = kf->features[correspondence.first].stereo_depth > 1e-6;
            isStereo2 = kf->features[correspondence.second].stereo_depth > 1e-6;

            double new_x = (kpu1.pt.x - kf->K(0, 2)) / kf->K(0, 0);
            double new_y = (kpu1.pt.y - kf->K(1, 2)) / kf->K(1, 1);
            Eigen::Vector3d normalised_first_point(new_x, new_y, 1.0); 

            new_x = (kpu2.pt.x - neighbour_kf->K(0, 2)) / neighbour_kf->K(0, 0);
            new_y = (kpu2.pt.y - neighbour_kf->K(1, 2)) / neighbour_kf->K(1, 1);
            Eigen::Vector3d normalised_second_point(new_x, new_y, 1.0);

            Eigen::Vector3d ray1 = kf->Tiw.rotationMatrix() *  normalised_first_point;
            Eigen::Vector3d ray2 = neighbour_kf->Tiw.rotationMatrix() * normalised_second_point;

            std::cout << ray1.norm() << " " << ray2.norm() << " sa fie norma 0 ar fi problematic\n";
            const float cosParallaxRays = ray1.dot(ray2) / (ray1.norm() * ray2.norm());

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            std::cout << kf->features[correspondence.first].depth << " " << neighbour_kf->features[correspondence.second].depth << " cele 2 adancimi\n";
            if(isStereo1)
                cosParallaxStereo1 = cos(2*atan2(1.6, kf->features[correspondence.first].depth ));
            else if(isStereo1)
                cosParallaxStereo2 = cos(2*atan2(1.6, neighbour_kf->features[correspondence.second].depth));

            cosParallaxStereo = std::min(cosParallaxStereo1,cosParallaxStereo2);

            std::cout << "merge tangenta\n";
            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (isStereo1 || isStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                std::cout << "ajunge sa faca triangularea liniara\n";
                Eigen::Matrix4d A;
                A.row(0) = normalised_first_point(0) *kf->Tiw.matrix3x4().row(2)-kf->Tiw.matrix3x4().row(0);
                A.row(1) = normalised_first_point(1) *kf->Tiw.matrix3x4().row(2)-kf->Tiw.matrix3x4().row(1);
                A.row(2) = normalised_second_point(0) * neighbour_kf->Tiw.matrix3x4().row(2) - neighbour_kf->Tiw.matrix3x4().row(0);
                A.row(3) = normalised_second_point(1) * neighbour_kf->Tiw.matrix3x4().row(2) - neighbour_kf->Tiw.matrix3x4().row(1);

                cv::Mat w,u,vt, A_mat;
                cv::eigen2cv(A, A_mat);
                cv::SVD::compute(A_mat,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
            }
            else if(isStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                Eigen::Vector4d out = kf->fromImageToWorld(correspondence.first);
                x3D = cv::Mat_<float>(3,1) << out(0), out(1), out(2);
            }
            else if(isStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                Eigen::Vector4d out = neighbour_kf->fromImageToWorld(correspondence.second);
                x3D = cv::Mat_<float>(3,1) << out(0), out(1), out(2);
            }
            else
                continue; //No stereo and very low parallax

            Eigen::Vector3d coordinates;
            cv::cv2eigen(x3D, coordinates);

            //Check triangulation in front of cameras
            float z1 = kf->Tiw.rotationMatrix().row(2).dot(coordinates) + kf->Tiw.translation().z();
            if(z1<=0)
                continue;

            float z2 = neighbour_kf->Tiw.rotationMatrix().row(2).dot(coordinates) + neighbour_kf->Tiw.translation().z();
            if(z2<=0)
                continue;
            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = pow(1.2, kpu1.octave);
            const float x1 = kf->Tiw.rotationMatrix().row(0).dot(coordinates)+kf->Tiw.translation().x();
            const float y1 = kf->Tiw.rotationMatrix().row(1).dot(coordinates)+kf->Tiw.translation().y();
            const float invz1 = 1.0/z1;

            if(!isStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kpu1.pt.x;
                float errY1 = v1 - kpu1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - 3.2 * invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kpu1.pt.x;
                float errY1 = v1 - kpu1.pt.y;
                float errX1_r = u1_r - kf->features[correspondence.first].stereo_depth;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pow(1.2, 2 * kpu2.octave);
            const float x2 = neighbour_kf->Tiw.rotationMatrix().row(0).dot(coordinates) + neighbour_kf->Tiw.translation().x();
            const float y2 = neighbour_kf->Tiw.rotationMatrix().row(1).dot(coordinates) + neighbour_kf->Tiw.translation().y();
            const float invz2 = 1.0/z2;
            if(!isStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kpu2.pt.x;
                float errY2 = v2 - kpu2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - 3.2 * invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kpu2.pt.x;
                float errY2 = v2 - kpu2.pt.y;
                float errX2_r = u2_r - neighbour_kf->features[correspondence.second].stereo_depth;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            Eigen::Vector3d normal1 = coordinates - kf->compute_camera_center();
            float dist1 = normal1.norm();

            Eigen::Vector3d normal2 = coordinates - neighbour_kf->compute_camera_center();
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = pow(1.2 , kpu1.octave - kpu2.octave);

            if(ratioDist* 1.8 < ratioOctave || ratioDist>ratioOctave * 1.8)
                continue;

            MapPoint* pMP = new MapPoint(kf, kpu1, kf->compute_camera_center(), Eigen::Vector4d(coordinates.x(), coordinates.y(), coordinates.z(), 1),
                     kf->orb_descriptors.row(correspondence.first));
               
            // TODO, de adaugat si observatia in frame-ul al doilea 
            kf->add_map_point(pMP, &kf->features[correspondence.first], pMP->orb_descriptor);
            neighbour_kf->add_map_point(pMP, &neighbour_kf->features[correspondence.second], pMP->orb_descriptor);
            pMP->add_observation_map_point(neighbour_kf, neighbour_kf->features[correspondence.second].descriptor, kf->compute_camera_center());
            // TODOOOOOO
            // (1) UPDATE_DEPTH
        }

        // std::vector<cv::DMatch> dmatches;
        
        // for (std::pair<int, int> correspondence : vMatchedIndices) {
        //     cv::DMatch dmatch;
        //     dmatch.queryIdx = correspondence.first;
        //     dmatch.trainIdx = correspondence.second;
        //     dmatch.distance = OrbMatcher::ComputeHammingDistance(kf->features[dmatch.queryIdx].descriptor, neighbour_kf->features[dmatch.trainIdx].descriptor); 
        //     dmatches.push_back(dmatch);
        // }
        // cv::Mat img_matches;
        // cv::drawMatches(kf->frame, kf->get_all_keypoints(), neighbour_kf->frame, neighbour_kf->get_all_keypoints(), dmatches, img_matches);
        // cv::imshow("Feature Matches", img_matches);
        // cv::waitKey(0);

    }
}


void LocalMapping::search_in_neighbours(KeyFrame *kf) {
    std::unordered_set<KeyFrame*> vpNeighKFs = mapp->get_local_keyframes(kf);
    for(KeyFrame* pKFi :  vpNeighKFs)
    {
        std::unordered_set<KeyFrame*> vpSecondNeighKFs = mapp->get_local_keyframes(pKFi);
        vpNeighKFs.insert(vpSecondNeighKFs.begin(), vpSecondNeighKFs.end());
    }

    // Search matches by projection from current KF in target KFs
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

    // Update points
    // vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    // for(MapPoint* pMP : kf->map_points)
    // {    
    //     pMP->ComputeDistinctiveDescriptors();
    //     pMP->UpdateNormalAndDepth();
    // }

    // Update connections in covisibility graph
    // mpCurrentKeyFrame->UpdateConnections();
}


Eigen::Matrix3d LocalMapping::compute_fundamental_matrix(KeyFrame *curr_kf, KeyFrame *neighbour_kf) {
    Eigen::Matrix3d R1w = curr_kf->Tiw.rotationMatrix();
    Eigen::Vector3d t1w = curr_kf->Tiw.translation();
    Eigen::Matrix3d R2w = neighbour_kf->Tiw.rotationMatrix();
    Eigen::Vector3d t2w = neighbour_kf->Tiw.translation();

    Eigen::Matrix3d R12 = R1w*R2w.transpose();
    Eigen::Vector3d t12 = -R12 * t2w + t1w;
 
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