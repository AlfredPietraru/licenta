#include "../include/BundleAdjustment.h"
#include "../include/rotation.h"

// AZI STUDIEZ LIBRARIA CERES
class BundleError
{
public:
    BundleError(Eigen::Vector3d observed, Eigen::Vector4<double> map_coordinate, double scale_sigma, Eigen::Matrix3d K,
                bool is_monocular) : observed(observed),
                                     map_coordinate(map_coordinate), scale_sigma(scale_sigma), K(K), is_monocular(is_monocular) {}

    template <typename T>
    bool operator()(const T *const angle_axis, const T *const t, T *residuals) const
    {
        T map_coordinate_world[3] = {T(map_coordinate(0)), T(map_coordinate(1)), T(map_coordinate(2))};
        T camera_coordinates[3];
        ceres::AngleAxisRotatePoint(angle_axis, map_coordinate_world, camera_coordinates);
        camera_coordinates[0] += t[0];
        camera_coordinates[1] += t[1];
        camera_coordinates[2] += t[2];
        T inv_d = T(1) / camera_coordinates[2];
        T x = K(0, 0) * camera_coordinates[0] * inv_d + K(0, 2);
        T y = K(1, 1) * camera_coordinates[1] * inv_d + K(1, 2);
        residuals[0] = (x - observed(0)) / scale_sigma;
        residuals[1] = (y - observed(1)) / scale_sigma;
        if (this->is_monocular) return true;

        T z_projected = K(0, 0) * (camera_coordinates[0] - BASELINE) * inv_d + K(0, 2);
        residuals[2] = (z_projected - observed(2)) / scale_sigma;
        return true;
    }

    static ceres::CostFunction *Create_Monocular(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 3, 3>(
            new BundleError(observed, map_coordinate, scale_sigma, K, true)));
    }

    static ceres::CostFunction *Create_Stereo(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 3, 3, 3>(
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

Sophus::SE3d BundleAdjustment::solve(KeyFrame *kf, std::unordered_map<MapPoint *, Feature*>& matches)
{
    const double BASELINE = 0.08;
    if (matches.size() == 0)
        return kf->Tiw;
    ceres::Problem problem;
    int nr_monocular_points = 0;
    int nr_stereo_points = 0;
    double angle_axis[3];
    
    angle_axis[0] = kf->Tiw.angleX();
    angle_axis[1] = kf->Tiw.angleY();
    angle_axis[2] = kf->Tiw.angleZ();
    double t[3];
    t[0] = kf->Tiw.translation().x();
    t[1] = kf->Tiw.translation().y();
    t[2] = kf->Tiw.translation().z();
    // std::cout << matches.size() << "\n";
    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        ceres::CostFunction *cost_function;
        cv::KeyPoint kp = it->second->get_key_point();
        double sigma = std::pow(1.2, kp.octave);
        // std::cout << sigma << " ";
        float dd = kf->compute_depth_in_keypoint(kp);
        if (dd <= 0)
        {
            nr_monocular_points++;
            cost_function = BundleError::Create_Monocular(Eigen::Vector3d(kp.pt.x, kp.pt.y, 0), it->first->wcoord, sigma, kf->K);
        }
        else
        {
            nr_stereo_points++;
            double fake_right_coordinate = kp.pt.x - (kf->K(0, 0) * BASELINE / dd);
            cost_function = BundleError::Create_Stereo(Eigen::Vector3d(kp.pt.x, kp.pt.y, fake_right_coordinate), it->first->wcoord, sigma, kf->K);
        }
        ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(cost_function, loss_function, angle_axis, t);
    }

    // std::cout << "au fost gasite atatea puncte monoculare " << nr_monocular_points << "\n";
    // std::cout << "au fost gasite atatea puncte stereo " << nr_stereo_points << "\n\n";
    // std::cout << "\n";
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    // options.linear_solver_ordering_type = ceres::AMD;
    // options.function_tolerance = 1e-7;
    // options.gradient_tolerance = 1e-7;
    // options.parameter_tolerance = 1e-8;
    // options.gradient_check_relative_precision = 1e-8;
    options.function_tolerance = 1e-5;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.use_inner_iterations = true;
    options.minimizer_progress_to_stdout = false;
    options.check_gradients = false;
    options.max_num_iterations = NUMBER_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
    Eigen::Vector3d angle_axis_eigen(angle_axis[0], angle_axis[1], angle_axis[2]);  
Eigen::AngleAxisd aa(angle_axis_eigen.norm(), angle_axis_eigen.normalized());
Eigen::Quaterniond quaternion(aa);
    return Sophus::SE3d(quaternion, Eigen::Vector3d(t[0], t[1], t[2]));
}