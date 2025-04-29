#include "../include/LocalMapping.h"


void LocalMapping::local_map(KeyFrame *kf) {
    mapp->add_new_keyframe(kf);
    if (this->first_kf) {
        this->recently_added = kf->map_points;
        this->first_kf = false;
    }
    this->map_points_culling(kf);
    int map_points_computed = this->compute_map_points(kf);
    if (map_points_computed == 0) std::cout << "NICIUN MAP POINT NU A FOST CALCULAT\n";
    std::cout << map_points_computed << " atatea map points noi adaugate\n";
    this->search_in_neighbours(kf);
    std::cout << "AICI INCEPE BA\n";
    bundleAdjustment->solve_ceres(this->mapp, kf);
    this->KeyFrameCulling(kf);
    // int value_with_problems = 0;
    // for (KeyFrame *kf : mapp->keyframes) {
    //     for (MapPoint *mp : kf->map_points) {
    //         if (!mp->find_keyframe(kf)) value_with_problems++;
    //     }        
    // }
    // std::cout << value_with_problems << " atatea puncte cheie au probleme\n";
    std::cout << "AICI SE TERMINA BA\n";

}


bool customComparison(KeyFrame *a, KeyFrame *b)
{
    return a->reference_idx < b->reference_idx; 
}

// DE ADAUGAT LOGICA PENTRU STERGERE O DATA LA 3 KF-uri;
void LocalMapping::map_points_culling(KeyFrame *curr_kf) {
    std::vector<MapPoint*> to_del;
    std::vector<MapPoint*> too_old_to_keep_looking;
    for (MapPoint *mp : this->recently_added) {
        if (mp->keyframes.size() == 0) {
            to_del.push_back(mp);
            continue;
        }
        double ratio = (double)mp->number_associations / mp->number_times_seen;
        if (ratio < 0.25f) {
            to_del.push_back(mp);
            continue;
        }

        if (mp->keyframes.size() == 3) {
            sort(mp->keyframes.begin(), mp->keyframes.end(), customComparison);
            bool first_two = (mp->keyframes[1]->reference_idx - mp->keyframes[0]->reference_idx) == 1;
            bool last_two = (mp->keyframes[2]->reference_idx - mp->keyframes[1]->reference_idx) == 1;
            bool last_equal_current_kf = mp->keyframes[2]->reference_idx == curr_kf->reference_idx; 

            if (first_two && last_two && last_equal_current_kf) {
                too_old_to_keep_looking.push_back(mp);
                continue;
            }
            to_del.push_back(mp);
            continue;
        }

        if (mp->keyframes.size() > 3) {
            too_old_to_keep_looking.push_back(mp);
            continue;
        }
    }
    for (MapPoint *mp : to_del) this->delete_map_point(mp);
    for (MapPoint *mp : too_old_to_keep_looking) this->recently_added.erase(mp);
}



bool is_coordinate_valid_for_keyframe(KeyFrame *kf, Feature *f, Eigen::Vector4d coordinates,  bool isStereo) {
    Eigen::Vector3d projected_on_kf = kf->Tcw.matrix3x4() * coordinates;
    const float z1 = projected_on_kf(2);
    if (z1 <= 1e-6) return false;
    const float invz1 = 1.0 / z1;
    const float &sigmaSquare1 = kf->POW_OCTAVE[f->get_undistorted_keypoint().octave] * kf->POW_OCTAVE[f->get_undistorted_keypoint().octave];
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
    std::unordered_set<KeyFrame*> keyframes = mapp->get_local_keyframes(kf);
    bool isStereo1 = false;
    bool isStereo2 = false;
    int nr_points_added = 0;
    bool was_addition_succesfull;
    for (KeyFrame* neighbour_kf : keyframes) {
        Eigen::Matrix3d fundamental_mat =  this->compute_fundamental_matrix(kf, neighbour_kf);
        std::vector<std::pair<int, int>> vMatchedIndices = OrbMatcher::search_for_triangulation(kf, neighbour_kf, fundamental_mat);
        for (std::pair<int, int> correspondence : vMatchedIndices) {
            Feature *f1 = &kf->features[correspondence.first];
            Feature *f2 = &neighbour_kf->features[correspondence.second];
            isStereo1 = f1->stereo_depth >= 1e-6;
            isStereo2 = f2->stereo_depth >= 1e-6;

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
            Eigen::Vector3d normal1 = coordinates_3d - kf->camera_center_world;
            float dist1 = normal1.norm();

            Eigen::Vector3d normal2 = coordinates_3d - neighbour_kf->camera_center_world;
            float dist2 = normal2.norm();

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = kf->POW_OCTAVE[f1->get_undistorted_keypoint().octave] / kf->POW_OCTAVE[f2->get_undistorted_keypoint().octave];

            if(ratioDist* 1.8 < ratioOctave || ratioDist>ratioOctave * 1.8)
                continue;

            MapPoint* pMP = new MapPoint(kf, f1->idx, f1->get_undistorted_keypoint(), kf->camera_center_world, coordinates,
                     kf->orb_descriptors.row(correspondence.first));
            this->recently_added.insert(pMP);
            pMP->increase_how_many_times_seen();
            pMP->increase_how_many_times_seen(); 
            was_addition_succesfull = Map::add_map_point_to_keyframe(kf, &kf->features[correspondence.first], pMP);
            if (!was_addition_succesfull) {
                std::cout << "NU S-A PUTUT REALIZA ADAUGAREA UNUI NOU MAP POINT IN LOCAL MAPPING\n";
                exit(1);
            }
            was_addition_succesfull = Map::add_map_point_to_keyframe(neighbour_kf, &neighbour_kf->features[correspondence.second], pMP);
            if (!was_addition_succesfull) {
                std::cout << "NU S-A PUTUT REALIZA ADAUGAREA UNUI NOU MAP POINT IN LOCAL MAPPING IN CELALALT FRAME\n";
                exit(1);
            }
            was_addition_succesfull = Map::add_keyframe_reference_to_map_point(pMP, &neighbour_kf->features[correspondence.second], neighbour_kf);
            if (!was_addition_succesfull) {
                std::cout << "NU S-A PUTUT ADAUGA REFERINTA PUNCTULUI IN KEYFRAME LOCAL MAPPING\n";
                continue;
            }
            nr_points_added++;
        }
    }
    return nr_points_added;
}

void LocalMapping::search_in_neighbours(KeyFrame *kf) {
    std::unordered_set<KeyFrame*> first_degree_neighbours = mapp->get_local_keyframes(kf);
    std::unordered_set<KeyFrame*> second_degree_neighbours;
    second_degree_neighbours.insert(first_degree_neighbours.begin(), first_degree_neighbours.end());
    for(KeyFrame* pKFi :  first_degree_neighbours)
    {
        std::unordered_set<KeyFrame*> vpSecondNeighKFs = mapp->get_local_keyframes(pKFi);
        second_degree_neighbours.insert(vpSecondNeighKFs.begin(), vpSecondNeighKFs.end());
    }

    second_degree_neighbours.erase(kf);
    for(KeyFrame *neighbour_kf : second_degree_neighbours)
    {
        OrbMatcher::Fuse(neighbour_kf, kf, 3);
        OrbMatcher::Fuse(kf, neighbour_kf, 3);
        bool result = mapp->update_graph_connections(kf, neighbour_kf);
        if (!result) {
            std::cout << "NU A REUSIT SA FACA UPDATE LA GRAPH\n";
            continue;
        }
    }
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
    bool was_deletion_succesfull;
    for (KeyFrame *kf : mp->keyframes) { 
        was_deletion_succesfull = Map::remove_map_point_from_keyframe(kf, mp);
        if (!was_deletion_succesfull) {
            std::cout << "NU S-A PUTUT SA STEARGA MAP POINT DIN FRAME-URI\n";
            continue;
        }
    }
    if (this->recently_added.find(mp) != this->recently_added.end()) this->recently_added.erase(mp);
    delete mp;
}


void LocalMapping::KeyFrameCulling(KeyFrame *kf)
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    std::unordered_set<KeyFrame*> local_keyframes = mapp->get_local_keyframes(kf);

    std::vector<KeyFrame*> to_delete_keyframes;
    for(KeyFrame *kf : local_keyframes)
    {
        if(kf->reference_idx == 0) continue;
        int nRedundantObservations=0;
        for(std::pair<MapPoint*, Feature*> map_point_feature : kf->mp_correlations)
        {
            MapPoint *mp = map_point_feature.first;
            Feature *f = map_point_feature.second;
            if (mp->keyframes.size() <= 3) continue; 
            const int &scaleLevel = f->kpu.octave;
            int nObs=0;
            for(KeyFrame *current_kf : mp->keyframes) 
            {
                if(current_kf == kf) continue;
                const int &scaleLeveli = current_kf->features[mp->data[current_kf]->feature_idx].get_undistorted_keypoint().octave;  
                if(scaleLeveli<=scaleLevel+1)
                {
                    nObs++;
                    if(nObs>=3) break;
                }
            }
            nRedundantObservations += (int)(nObs >= 3);
        }  
        // std::cout << nRedundantObservations <<  " " << kf->map_points.size() << " atatea observatii redundante intalnite\n";
        if(nRedundantObservations > 0.9 * kf->map_points.size())
            to_delete_keyframes.push_back(kf);
    }
    std::cout << to_delete_keyframes.size() << " aceste keyframe-uri ar putea fi sterse to_delete_keyframes\n";
    // for (KeyFrame *kf : to_delete_keyframes) {
    //     // aici voi face stergerea
    // }
}