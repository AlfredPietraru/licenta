#include "BundleAdjustment.h"

class BundleError
{
    const double FOCAL_LENGTH = 500.0;
    const double X_CAMERA_OFFSET = 325.1;
    const double Y_CAMERA_OFFSET = 250.7;

public:
    BundleError(double observation_x, double observation_y) : observed_x(observation_x),
                                                              observed_y(observation_y) {}

    template <typename T>
    bool operator()(const T *const pose, const T *const point, T *residuals) const
    {
        Eigen::Matrix4<T> eigen_pose; 
        Eigen::Vector4<T> eigen_point;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                eigen_pose(i, j) = pose[i * 3 + j];
            }
        }
        for (int i = 0; i < 3; i++) {
            eigen_pose(i, 3) = pose[9 + i];
        }
        for (int i = 0; i < 2; i++) {
            eigen_pose(3, i) = (T)0;
        }
        eigen_pose(3, 3) = (T)1;
        for (int i = 0; i < 4; i++) {
            eigen_point(i) = point[i];
        }

        const Eigen::Vector4<T> camera_coordinates = eigen_pose * eigen_point;
        T d = camera_coordinates(2);
        T x = FOCAL_LENGTH * camera_coordinates(0) / d + X_CAMERA_OFFSET;
        T y = FOCAL_LENGTH * camera_coordinates(1) / d + Y_CAMERA_OFFSET;
        residuals[0] = observed_x - x;
        residuals[1] = observed_y - y;
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 12, 4>(
            new BundleError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

BundleAdjustment::BundleAdjustment(std::vector<MapPoint> map_points, std::vector<cv::KeyPoint> kps, Eigen::Matrix4d T) {
    this->T = T;
    this->map_points = map_points;
    this->kps = kps;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            this->data_T[i * 3 + j] = T(i, j);
        }
    }
    this->data_T[9] = T(0, 3);
    this->data_T[10] = T(1, 3);
    this->data_T[11] = T(2, 3);
    for (int i = 0; i < map_points.size(); i++) {
        this->poins_coordinates.push_back(map_points[i].wcoord.data());
    }
}

Eigen::Matrix4d BundleAdjustment::return_optimized_pose() {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            T(i, j) = this->data_T[i * 3 + j];
        }
    }
    for (int i = 0; i < 3; i++) {
        T(i, 3) = this->data_T[9 + i];
    }
    return T;
}

void BundleAdjustment::solve() {
    ceres::Problem problem;
    for (int i = 0; i < this->map_points.size(); i++) {
        ceres::CostFunction *cost_function;
        cost_function = BundleError::Create(kps[i].pt.x, kps[i].pt.y);
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost_function, loss_function, this->data_T, this->map_points[i].wcoord.data());
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}