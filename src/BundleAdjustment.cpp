#include "../include/BundleAdjustment.h"
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
        if (this->is_monocular) return true;

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

Sophus::SE3d BundleAdjustment::solve(KeyFrame *kf, std::unordered_map<MapPoint *, Feature*>& matches, int maximum_selected)
{
    const double BASELINE = 0.08;
    if (matches.size() == 0) return kf->Tiw;
    ceres::Problem problem;
    Eigen::Vector3d angle_axis_sophus = kf->Tiw.so3().log();
    double pose_parameters[6];
    pose_parameters[0] = angle_axis_sophus[0]; 
    pose_parameters[1] = angle_axis_sophus[1];
    pose_parameters[2] = angle_axis_sophus[2];
    pose_parameters[3] = kf->Tiw.translation().x();
    pose_parameters[4] = kf->Tiw.translation().y();
    pose_parameters[5] = kf->Tiw.translation().z();

    int nr_selected = 0;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    for (auto it = matches.begin(); it != matches.end(); it++)
    {   
        ceres::CostFunction *cost_function;
        cv::KeyPoint kp = it->second->get_key_point();
        double sigma = std::pow(1.2, kp.octave);
        if (it->second->depth <= 0)
        {
            cost_function = BundleError::Create_Monocular(Eigen::Vector3d(kp.pt.x, kp.pt.y, 0), it->first->wcoord, sigma, kf->K);
        }
        else
        {
            double fake_right_coordinate = kp.pt.x - (kf->K(0, 0) * BASELINE / it->second->depth);
            cost_function = BundleError::Create_Stereo(Eigen::Vector3d(kp.pt.x, kp.pt.y, fake_right_coordinate), it->first->wcoord, sigma, kf->K);
        }
        problem.AddResidualBlock(cost_function, loss_function, pose_parameters);
        nr_selected ++;
        if (nr_selected == maximum_selected) break;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_SCHUR;
    options.function_tolerance = 1e-7;
    options.gradient_tolerance = 1e-7;
    options.parameter_tolerance = 1e-8;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.check_gradients = true;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = NUMBER_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
    Eigen::Vector3d angle_axis_eigen(pose_parameters[0], pose_parameters[1], pose_parameters[2]);  
    Eigen::AngleAxisd aa(angle_axis_eigen.norm(), angle_axis_eigen.normalized());
    Eigen::Quaterniond quaternion(aa);
    Eigen::Quaterniond old_quaternion = kf->Tiw.unit_quaternion();
    if (old_quaternion.dot(quaternion) < 0) {
        quaternion.coeffs() *= -1;
    }
    return Sophus::SE3d(quaternion, Eigen::Vector3d(pose_parameters[3], pose_parameters[4], pose_parameters[5]));
}