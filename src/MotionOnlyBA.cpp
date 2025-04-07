#include "../include/MotionOnlyBA.h"
#include "../include/rotation.h"

class BundleError
{
public:
    BundleError(Eigen::Vector3d observed, Eigen::Vector4<double> map_coordinate, double scale_sigma, Eigen::Matrix3d K,
                bool is_monocular) : observed(observed),
                                     map_coordinate(map_coordinate), scale_sigma(scale_sigma), K(K), is_monocular(is_monocular) {}

    template <typename T>
    bool operator()(const T *const params, T *residuals) const
    {
        T map_coordinate_world[3] = {T(map_coordinate(0)), T(map_coordinate(1)), T(map_coordinate(2))};
        T camera_coordinates[3];
        ceres::AngleAxisRotatePoint(params, map_coordinate_world, camera_coordinates);
        camera_coordinates[0] += params[3];
        camera_coordinates[1] += params[4];
        camera_coordinates[2] += params[5];
        T inv_d = T(1) / (camera_coordinates[2] + T(1e-6));
        T x = T(K(0, 0)) * camera_coordinates[0] * inv_d + T(K(0, 2));
        T y = T(K(1, 1)) * camera_coordinates[1] * inv_d + T(K(1, 2));
        residuals[0] = (x - observed(0)) / scale_sigma;
        residuals[1] = (y - observed(1)) / scale_sigma;
        if (this->is_monocular)
            return true;

        T z_projected = x - T(K(0, 0)) * BASELINE * inv_d;
        residuals[2] = (z_projected - observed(2)) / scale_sigma;
        return true;
    }

    static ceres::CostFunction *Create_Monocular(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 6>(
            new BundleError(observed, map_coordinate, scale_sigma, K, true)));
    }

    static ceres::CostFunction *Create_Stereo(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 3, 6>(
            new BundleError(observed, map_coordinate, scale_sigma, K, false)));
    }

private:
    const double BASELINE = 0.08;
    bool is_monocular;
    Eigen::Matrix3d K;
    Eigen::Vector3d observed;
    Eigen::Vector4d map_coordinate;
    double scale_sigma;
};

double get_rgbd_reprojection_error(MapPoint *mp, Eigen::Matrix3d K, Sophus::SE3d pose, Feature* feature, double chi2) {
    double residuals[3];
    Eigen::Vector3d camera_coordinates = pose.matrix3x4() * mp->wcoord;
    if (camera_coordinates[2] <= 1e-6) return 100000;
    double inv_d = 1 / camera_coordinates[2];
    double x = K(0, 0) * camera_coordinates[0] * inv_d + K(0, 2);
    double y = K(1, 1) * camera_coordinates[1] * inv_d + K(1, 2);
    cv::KeyPoint kp = feature->get_key_point();
    residuals[0] = (x - kp.pt.x) / std::pow(1.2, kp.octave);
    residuals[1] = (y - kp.pt.y) / std::pow(1.2, kp.octave);
    double z_projected = K(0, 0) * (camera_coordinates[0] - 0.08) * inv_d + K(0, 2);
    residuals[2] = (z_projected - feature->stereo_depth) / std::pow(1.2, kp.octave);
    double out = residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2];
    if (out <= chi2) return out;
    return 2 * sqrt(chi2) * sqrt(out) - chi2;
}


double get_monocular_reprojection_error(MapPoint *mp, Eigen::Matrix3d K, Sophus::SE3d pose, Feature* feature, double chi2) {
    double residuals[2];
    Eigen::Vector3d camera_coordinates = pose.matrix3x4() * mp->wcoord;
    if (camera_coordinates[2] <= 1e-6) return 100000;
    double inv_d = 1 / camera_coordinates[2];
    double x = K(0, 0) * camera_coordinates[0] * inv_d + K(0, 2);
    double y = K(1, 1) * camera_coordinates[1] * inv_d + K(1, 2);
    cv::KeyPoint kp = feature->get_key_point();
    residuals[0] = (x - kp.pt.x) / std::pow(1.2, kp.octave);
    residuals[1] = (y - kp.pt.y) / std::pow(1.2, kp.octave);
    double out = residuals[0] * residuals[0] + residuals[1] * residuals[1];
    if (out <= chi2) return out;
    return 2 * sqrt(chi2) * sqrt(out) - chi2;
}



Sophus::SE3d compute_pose(KeyFrame *kf, double *pose_parameters) {
    Eigen::Vector3d angle_axis_eigen(pose_parameters[0], pose_parameters[1], pose_parameters[2]);
    Eigen::AngleAxisd aa(angle_axis_eigen.norm(), angle_axis_eigen.normalized());
    Eigen::Quaterniond quaternion(aa);
    Eigen::Quaterniond old_quaternion = kf->Tiw.unit_quaternion();
    if (old_quaternion.dot(quaternion) < 0)
    {
        quaternion.coeffs() *= -1;
    }
    return Sophus::SE3d(quaternion, Eigen::Vector3d(pose_parameters[3], pose_parameters[4], pose_parameters[5]));
}

Sophus::SE3d BundleAdjustment::solve(KeyFrame *kf, std::unordered_map<MapPoint *, Feature *> &matches)
{
    if (matches.size() < 3)
        return kf->Tiw;

    Sophus::SE3d pose = kf->Tiw;
    const double BASELINE = 0.08;
    Eigen::Vector3d angle_axis_sophus = pose.so3().log();
    double pose_parameters[6];
    pose_parameters[0] = angle_axis_sophus[0];
    pose_parameters[1] = angle_axis_sophus[1];
    pose_parameters[2] = angle_axis_sophus[2];
    pose_parameters[3] = pose.translation().x();
    pose_parameters[4] = pose.translation().y();
    pose_parameters[5] = pose.translation().z();

    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    for (int i = 0; i < 4; i++)
    {
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
        options.function_tolerance = 1e-7;
        options.gradient_tolerance = 1e-7;
        options.parameter_tolerance = 1e-8;
        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.check_gradients = false;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 10;
        
        ceres::LossFunction *loss_function_mono = new ceres::HuberLoss(sqrt(chi2Mono[i]));
        ceres::LossFunction *loss_function_stereo = new ceres::HuberLoss(sqrt(chi2Stereo[i]));
        
        std::unordered_map<MapPoint*, ceres::ResidualBlockId> residual_rgbd_points;
        std::unordered_map<MapPoint*, ceres::ResidualBlockId> residual_monocular_points;
        int inlier = 0;
        for (auto it = matches.begin(); it != matches.end(); it++)
        {
            MapPoint *mp = it->first;
            if (kf->check_map_point_outlier(mp))  continue;
            ceres::CostFunction *cost_function;
            cv::KeyPoint kp = it->second->get_key_point();
            if (it->second->stereo_depth <= 1e-6)
            {
                cost_function = BundleError::Create_Monocular(Eigen::Vector3d(kp.pt.x, kp.pt.y, 0), it->first->wcoord,
                                                              std::pow(1.2, kp.octave), kf->K);
                ceres::ResidualBlockId id = problem.AddResidualBlock(cost_function, loss_function_mono, pose_parameters);
                residual_monocular_points.insert({it->first, id});
            }
            else
            {
                cost_function = BundleError::Create_Stereo(Eigen::Vector3d(kp.pt.x, kp.pt.y, it->second->stereo_depth), it->first->wcoord,
                                                           std::pow(1.2, kp.octave), kf->K);    
                ceres::ResidualBlockId id = problem.AddResidualBlock(cost_function, loss_function_stereo, pose_parameters);
                residual_rgbd_points.insert({it->first, id});
            }
       }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.FullReport() << "\n";        

        // check outlier values
        Sophus::SE3d intermediate_pose = compute_pose(kf, pose_parameters);
        for (auto it = matches.begin(); it != matches.end(); it++) {
            MapPoint *mp = it->first;
            if (mp == nullptr) continue;
            if (residual_monocular_points.find(mp) != residual_monocular_points.end()) {
                double residual[2];
                problem.EvaluateResidualBlock(residual_monocular_points[mp], true, nullptr, residual, nullptr);
                const float error = residual[0] * residual[0] + residual[1] * residual[1];    
                // std::cout << error << " ";
                // std::cout << residual[0] << " " << residual[1] << "     ";
                if(error > chi2Mono[i]) {
                    kf->add_outlier_element(mp);
                }  else {
                    inlier++;
                }
                continue;
            }
            if (residual_rgbd_points.find(mp) != residual_rgbd_points.end()) {
                double residual[3];
                problem.EvaluateResidualBlock(residual_rgbd_points[mp], true, nullptr, residual, nullptr);
                const float error = residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2];
                // std::cout << error << " ";
                // std::cout << residual[0] << " " << residual[1] << " "  << residual[2] << "     ";
                if(error > chi2Stereo[i]) {
                    kf->add_outlier_element(mp);
                } else {
                    inlier++;
                }
                continue;
            }
            if (it->second->stereo_depth <= 0) {
                double error = get_monocular_reprojection_error(mp, kf->K, intermediate_pose, it->second, chi2Mono[i]);
                // std::cout << error << " ";
                if(error > chi2Mono[i]) {
                    kf->add_outlier_element(mp);
                } else {
                    inlier++;
                } 
            } else {
                double error = get_rgbd_reprojection_error(mp, kf->K, intermediate_pose, it->second, chi2Stereo[i]);
                // std::cout << error << " ";
                if (error > chi2Stereo[i]) {
                    kf->add_outlier_element(mp);
                } else {
                    inlier++;
                } 
            }            
        }
        // std::cout << inlier << " atatea inliere gasite la epoca " << i << "\n";
    }
    return compute_pose(kf, pose_parameters);    
}