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
        T inv_d = T(1) / camera_coordinates[2];
        T x = T(K(0, 0)) * camera_coordinates[0] * inv_d + T(K(0, 2));
        T y = T(K(1, 1)) * camera_coordinates[1] * inv_d + T(K(1, 2));
        residuals[0] = (x - observed(0)) / scale_sigma;
        residuals[1] = (y - observed(1)) / scale_sigma;
        if (this->is_monocular)
            return true;

        T z_projected = camera_coordinates[0] - T(K(0, 0)) * BASELINE * inv_d;
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
        options.check_gradients = true;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 10;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        std::unordered_map<ceres::ResidualBlockId, MapPoint*> residual_rgbd_points;
        std::unordered_map<ceres::ResidualBlockId, MapPoint*> residual_monocular_points;
        for (auto it = matches.begin(); it != matches.end(); it++)
        {

            if (it->first->is_outlier)
                continue;
            ceres::CostFunction *cost_function;
            cv::KeyPoint kp = it->second->get_key_point();
            if (it->second->stereo_depth <= 0)
            {
                cost_function = BundleError::Create_Monocular(Eigen::Vector3d(kp.pt.x, kp.pt.y, 0), it->first->wcoord,
                                                              std::pow(1.2, kp.octave), kf->K);
                ceres::ResidualBlockId id = problem.AddResidualBlock(cost_function, loss_function, pose_parameters);
                residual_monocular_points.insert({id, it->first});
            }
            else
            {
                cost_function = BundleError::Create_Stereo(Eigen::Vector3d(kp.pt.x, kp.pt.y, it->second->stereo_depth), it->first->wcoord,
                                                           std::pow(1.2, kp.octave), kf->K);    
                ceres::ResidualBlockId id = problem.AddResidualBlock(cost_function, loss_function, pose_parameters);
                residual_rgbd_points.insert({id, it->first});
            }
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.FullReport() << "\n";        

        // check outlier values
        for (auto it = residual_monocular_points.begin(); it != residual_monocular_points.end(); it++) {
            double residual[2];
            problem.EvaluateResidualBlock(it->first, true, nullptr, residual, nullptr);
            const float error = residual[0] * residual[0] + residual[1] * residual[1];
            if (error > chi2Mono[i])  {
                it->second->is_outlier = true;
            }

        }
        for (auto it = residual_rgbd_points.begin(); it != residual_rgbd_points.end(); it++) {
            double residual[3];
            problem.EvaluateResidualBlock(it->first, true, nullptr, residual, nullptr);
            const float error = residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2];
            if (error > chi2Stereo[i]) {
                it->second->is_outlier = true;
            }
        }
    }

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