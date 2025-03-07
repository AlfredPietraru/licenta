#include "../include/BundleAdjustment.h"
#include "../include/rotation.h"


class BundleError
{
    const double FOCAL_LENGTH = 500.0;
    const double X_CAMERA_OFFSET = 325.1;
    const double Y_CAMERA_OFFSET = 250.7;

public:
    BundleError(double observation_x, double observation_y, Eigen::Vector4<double> map_coordinate) : observed_x(observation_x),
                                                              observed_y(observation_y), map_coordinate(map_coordinate), scale_sigma(scale_sigma) {}

    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        Eigen::Vector3<T> translation;
        for (int i = 0; i < 3; i++) {
            translation(i) = se3[i + 4];
        }
        // problema cu quaternionii, imi dadeau cu minus unele elemente de pe diagonala principala 
        Eigen::Quaternion<T> q(se3[3], se3[0], se3[1], se3[2]); 
        q.normalize();
        Sophus::SE3<T> pose = Sophus::SE3<T>(q, translation);
        
        const Eigen::Vector3<T> camera_coordinates = pose.matrix3x4() * map_coordinate;
        T d = camera_coordinates(2);
        // if (d < 0) return false;
        T x = FOCAL_LENGTH * camera_coordinates(0) / d + X_CAMERA_OFFSET;
        T y = FOCAL_LENGTH * camera_coordinates(1) / d + Y_CAMERA_OFFSET;
        // residuals[0] = (observed_x - x) / scale_sigma;
        // residuals[1] = (observed_y - y) / scale_sigma;
        residuals[0] = (x - observed_x);
        residuals[1] = (y - observed_y); 
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y, Eigen::Vector4<double> map_coordinate, double scale_sigma)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 6>(
            new BundleError(observed_x, observed_y, map_coordinate)));
    }

private:
    double observed_x;
    double observed_y;
    Eigen::Vector4<double> map_coordinate;
    double scale_sigma;
};

Sophus::SE3d BundleAdjustment::solve(Sophus::SE3d T, std::vector<MapPoint*> map_points, std::vector<cv::KeyPoint> kps)
{ 
    // std::cout << T.matrix() << "\n";
    // Eigen::Vector3d test_translation;
    // for (int i = 0; i < 3; i++) {
    //     test_translation(i) = T.data()[i + 4];
    // }
    // Eigen::Quaterniond test_q(T.data()[3], T.data()[0], T.data()[1], T.data()[2]); 
    // test_q.normalize();
    // T = Sophus::SE3d(test_q, test_translation);
    // std::cout << T.matrix() << "\n";

    double *data = T.data();
    ceres::Problem problem;
    if (map_points.size() == 0) return T;
    for (int i = 0; i < map_points.size(); i++)
    {
        ceres::CostFunction *cost_function;
        double sigma = std::pow(1.2, kps[i].octave);
        std::cout << sigma << " ";
        cost_function = BundleError::Create(kps[i].pt.x, kps[i].pt.y, map_points[i]->wcoord, sigma);
        ceres::LossFunction *loss_function = new ceres::HuberLoss(HUBER_LOSS_VALUE);
        problem.AddResidualBlock(cost_function, loss_function, data);
    }
    std::cout << "\n";
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = NUMBER_ITERATIONS;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    Eigen::Vector3d translation;
    for (int i = 0; i < 3; i++) {
        translation(i) = data[i + 4];
    }
    Eigen::Quaterniond q(data[3], data[0], data[1], data[2]); 
    q.normalize();
    T = Sophus::SE3d(q, translation);
    std::cout << T.matrix() << "\n";
    return T;
    
}