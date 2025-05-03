#include "../include/BundleAdjustment.h"
using SE3Manifold = ceres::ProductManifold<ceres::QuaternionManifold, ceres::EuclideanManifold<3>>;

double get_rgbd_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* f, double chi2) {
    double residuals[3];
    Eigen::Vector3d camera_coordinates = kf->Tcw.matrix3x4() * mp->wcoord;
    if (camera_coordinates[2] <= 1e-6) return 1e-6;
    double inv_d = 1 / camera_coordinates[2];
    double x = kf->K(0, 0) * camera_coordinates[0] * inv_d + kf->K(0, 2);
    double y = kf->K(1, 1) * camera_coordinates[1] * inv_d + kf->K(1, 2);
    residuals[0] = (x - f->kpu.pt.x) / kf->POW_OCTAVE[f->kpu.octave];
    residuals[1] = (y - f->kpu.pt.y) / kf->POW_OCTAVE[f->kpu.octave];
    double z_projected = x - kf->K(0, 0) * 0.08 * inv_d;
    residuals[2] = (z_projected - f->right_coordinate) / kf->POW_OCTAVE[f->kpu.octave];
    double a = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]);
    if (a <= sqrt(chi2)) return pow(a, 2) / 2;
    return sqrt(chi2) * a - chi2 / 2;
}

double get_monocular_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* f, double chi2) {
    double residuals[2];
    Eigen::Vector3d camera_coordinates = kf->Tcw.matrix3x4() * mp->wcoord;
    if (camera_coordinates[2] <= 1e-6) return 1e-6;
    double inv_d = 1 / camera_coordinates[2];
    double x = kf->K(0, 0) * camera_coordinates[0] * inv_d + kf->K(0, 2);
    double y = kf->K(1, 1) * camera_coordinates[1] * inv_d + kf->K(1, 2);
    residuals[0] = (x - f->kpu.pt.x) / kf->POW_OCTAVE[f->kpu.octave];
    residuals[1] = (y - f->kpu.pt.y) / kf->POW_OCTAVE[f->kpu.octave];
    double a = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1]);
    if (a <= sqrt(chi2)) return pow(a, 2) / 2;
    return sqrt(chi2) * a - chi2 / 2;
}


void execute_problem(std::unordered_set<KeyFrame*>& local_keyframes, std::unordered_set<KeyFrame*> &fixed_keyframes,
        std::unordered_set<MapPoint*>& local_map_points, int number_iterations, bool to_output) {
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = to_output;
    options.function_tolerance = 1e-7;
    options.gradient_tolerance = 1e-7;
    options.parameter_tolerance = 1e-8;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = number_iterations;
    double chi2Mono = 5.991;
    double chi2Stereo = 7.815;
    ceres::LossFunction *loss_function_mono = new ceres::HuberLoss(sqrt(chi2Mono));
    ceres::LossFunction *loss_function_stereo = new ceres::HuberLoss(sqrt(chi2Stereo));

    for (KeyFrame *kf : local_keyframes) {
        double* pose = kf->pose_vector;
        double norm = sqrt(pose[0]*pose[0] + pose[1]*pose[1] + pose[2]*pose[2] + pose[3]*pose[3]);
        assert(std::abs(norm - 1.0) < 1e-3);
        problem.AddParameterBlock(kf->pose_vector, 7);
        problem.SetManifold(kf->pose_vector, new SE3Manifold());
    }

    for (MapPoint *mp : local_map_points) {
        for (KeyFrame *kf : mp->keyframes) {
            Feature *f = kf->mp_correlations[mp];
            bool is_local_keyframe = local_keyframes.find(kf) != local_keyframes.end();
            bool is_fixed_keyframe = fixed_keyframes.find(kf) != fixed_keyframes.end();
            bool is_monocular = f->right_coordinate <= 1e-6;
            if (is_local_keyframe && is_monocular) {
                ceres::CostFunction *cost_function = BundleAdjustmentError::Create_Variable_Monocular(kf, f);
                problem.AddResidualBlock(cost_function, loss_function_mono, kf->pose_vector, mp->wcoord_3d.data());
                continue;
            }
            if (is_local_keyframe && !is_monocular) {
                ceres::CostFunction *cost_function = BundleAdjustmentError::Create_Variable_Stereo(kf, f);
                problem.AddResidualBlock(cost_function, loss_function_stereo, kf->pose_vector, mp->wcoord_3d.data());
                continue;
            }
            if (is_fixed_keyframe && is_monocular) {
                ceres::CostFunction *cost_function = BundleAdjustmentError::Create_Static_Monocular(kf, f);
                problem.AddResidualBlock(cost_function, loss_function_mono, mp->wcoord_3d.data());
                continue;
            }
            if (is_fixed_keyframe && !is_monocular) {
                ceres::CostFunction *cost_function = BundleAdjustmentError::Create_Static_Stereo(kf, f);
                problem.AddResidualBlock(cost_function, loss_function_stereo, mp->wcoord_3d.data());
                continue;
            }
        }
    }
    
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (to_output) {
        std::cout << summary.FullReport() << "\n";
    }
}

void restore_computed_values(std::unordered_set<KeyFrame*>& local_keyframes, std::unordered_set<MapPoint*>& local_map_points) {
    for (KeyFrame *kf : local_keyframes) {
        kf->set_keyframe_position(kf->compute_pose());
    }    
    for (MapPoint *mp : local_map_points) {
        mp->update_map_point_coordinate();
    }
}

void BundleAdjustment::solve_ceres(Map *mapp, KeyFrame *frame) {
    if (mapp->keyframes.size() == 1) return;
   std::unordered_set<KeyFrame*> local_keyframes = mapp->get_local_keyframes(frame);
    if (local_keyframes.size() == 0) {
        std::cout << "CEVA E GRESIT IN BUNDLE ADJUSTMENT ACEST KEYFRAME ESTE IZOLAT\n";
        return;
    }
    local_keyframes.insert(frame);
    for (KeyFrame *kf : local_keyframes) {
        if (kf->reference_idx == 0) {
            local_keyframes.erase(kf);
            break;
        } 
    }


    std::unordered_set<MapPoint*> local_map_points;
    for (KeyFrame *kf : local_keyframes) {
        local_map_points.insert(kf->map_points.begin(), kf->map_points.end());
    }

    if (local_map_points.size() == 0) {
        std::cout << "CEVA NU E BINE NU EXISTA DELOC PUNCTE IN LOCAL MAP PENTRU OPTIMIZARE\n";
        return;
    }

    std::unordered_set<KeyFrame*> fixed_keyframes;
    for (MapPoint *mp : local_map_points) {
        for (KeyFrame *kf_which_sees_map_point : mp->keyframes) {
            if (local_keyframes.find(kf_which_sees_map_point) == local_keyframes.end()) fixed_keyframes.insert(kf_which_sees_map_point);
        }
    }

    execute_problem(local_keyframes, fixed_keyframes, local_map_points, 5, false);
    restore_computed_values(local_keyframes, local_map_points);
    double chi2Mono = 5.991;
    double chi2Stereo = 7.815;

    std::vector<MapPoint *> to_delete; 
    for (MapPoint *mp : local_map_points) {
        std::vector<KeyFrame*> copy_keyframe_vector(mp->keyframes.begin(), mp->keyframes.end());
        for (KeyFrame *kf : copy_keyframe_vector) {
            Feature *f = kf->mp_correlations[mp];
            bool is_outlier;
            if (f->right_coordinate <= 1e-6) {
                double error = get_monocular_reprojection_error(kf, mp, f, chi2Mono);
                is_outlier = error > chi2Mono;
            } else {
                double error = get_rgbd_reprojection_error(kf, mp, f, chi2Stereo);
                is_outlier = error > chi2Stereo;
            }
            if (is_outlier) {
                Map::remove_map_point_from_keyframe(kf, mp);
                Map::remove_keyframe_reference_from_map_point(mp, kf);
            }
        }
        if (mp->keyframes.size() == 0) {
            to_delete.push_back(mp);
        }
    }
    // std::cout << to_delete.size() << " ATATEA PUNCTE AU FOST GASITE DE STERS\n";
    for (MapPoint *mp : to_delete) {
        local_map_points.erase(mp);
    }

    execute_problem(local_keyframes, fixed_keyframes, local_map_points, 10, false);
    restore_computed_values(local_keyframes, local_map_points);
}