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
    bool operator()(const T *const q, const T *const t, T *residuals) const
    {
        T map_coordinate_world[3];
        map_coordinate_world[0] = (T)map_coordinate(0);
        map_coordinate_world[1] = (T)map_coordinate(1);
        map_coordinate_world[2] = (T)map_coordinate(2);
        T camera_coordinates[3];
        ceres::QuaternionRotatePoint(q, map_coordinate_world, camera_coordinates);
        camera_coordinates[0] += t[0];
        camera_coordinates[1] += t[1];
        camera_coordinates[2] += t[2];
        T d = camera_coordinates[2];

        if (d < T(1e-6)) {
            residuals[0] = T(1e4); 
            residuals[1] = T(1e4);
            if (this->is_monocular) return true;
            residuals[2] = T(1e4);
            return true;
        }

        T x = K(0, 0) * camera_coordinates[0] / d + K(0, 2);
        T y = K(1, 1) * camera_coordinates[1] / d + K(1, 2);
        residuals[0] = (x - observed(0)) / scale_sigma;
        residuals[1] = (y - observed(1)) / scale_sigma;
        if (this->is_monocular) return true;

        T z_projected = K(0, 0) * (camera_coordinates[0] - BASELINE) / d + K(0, 2);
        residuals[2] = (z_projected - observed(2)) / scale_sigma;  
        return true;
    }

    static ceres::CostFunction *Create_Monocular(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::NumericDiffCostFunction<BundleError, ceres::RIDDERS, 2, 4, 3>(
            new BundleError(observed, map_coordinate, scale_sigma, K, true)));
    }

    static ceres::CostFunction *Create_Stereo(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::NumericDiffCostFunction<BundleError, ceres::RIDDERS, 3, 4, 3>(
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

Sophus::SE3d BundleAdjustment::solve(KeyFrame *kf, std::unordered_map<MapPoint *, Feature*> matches)
{ 
    const double BASELINE = 0.08;
    if (matches.size() == 0) return kf->Tiw;
    ceres::Problem problem;
    int nr_monocular_points = 0;
    int nr_stereo_points = 0;
    double q[4];
    q[0] = kf->Tiw.unit_quaternion().w();
    q[1] = kf->Tiw.unit_quaternion().x();
    q[2] = kf->Tiw.unit_quaternion().y();
    q[3] = kf->Tiw.unit_quaternion().z();
    double t[3];
    t[0] = kf->Tiw.translation().x();
    t[1] = kf->Tiw.translation().y();
    t[2] = kf->Tiw.translation().z();

    for (auto it = matches.begin(); it != matches.end(); it++)
    {
        ceres::CostFunction *cost_function;
        cv::KeyPoint kp = it->second->get_key_point();
        double sigma = std::pow(1.2, kp.octave);
        // std::cout << sigma << " ";
        float dd = kf->compute_depth_in_keypoint(kp);
        if (dd <= 0) {
            nr_monocular_points ++;
            cost_function = BundleError::Create_Monocular(Eigen::Vector3d(kp.pt.x, kp.pt.y, 0), it->first->wcoord, sigma, kf->K);
        } else {
            nr_stereo_points++;
            double fake_right_coordinate = kp.pt.x - (kf->K(0, 0) * BASELINE / dd);
            cost_function = BundleError::Create_Stereo(Eigen::Vector3d(kp.pt.x, kp.pt.y, fake_right_coordinate), it->first->wcoord, sigma, kf->K);
        }
        ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
        problem.AddResidualBlock(cost_function, loss_function, q, t);
    }
    std::cout << "au fost gasite atatea puncte monoculare " << nr_monocular_points << "\n";
    std::cout << "au fost gasite atatea puncte stereo " << nr_stereo_points << "\n\n";
    std::cout << "\n";
    ceres::Solver::Options options;  
    options.linear_solver_type =  ceres::LinearSolverType::SPARSE_NORMAL_CHOLESKY;
    // options.linear_solver_ordering_type = ceres::AMD;
    options.function_tolerance = 1e-7;
    options.gradient_tolerance = 1e-7;
    options.parameter_tolerance = 1e-8;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.check_gradients = true;
    options.max_num_iterations = NUMBER_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    return Sophus::SE3d(Eigen::Quaterniond(q[0], q[1], q[2], q[3]), Eigen::Vector3d(t[0], t[1], t[2]));
}