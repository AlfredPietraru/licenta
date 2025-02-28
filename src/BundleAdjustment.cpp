#include "../include/BundleAdjustment.h"
#include "../include/rotation.h"

class BundleError
{
    const double FOCAL_LENGTH = 500.0;
    const double X_CAMERA_OFFSET = 325.1;
    const double Y_CAMERA_OFFSET = 250.7;

public:
    BundleError(double observation_x, double observation_y, Eigen::Vector4<double> map_coordinate) : observed_x(observation_x),
                                                              observed_y(observation_y), map_coordinate(map_coordinate) {}

    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        Sophus::SE3<T> pose = Sophus::SE3<T>::exp(Eigen::Map<const Eigen::Matrix<T, 6, 1>>(se3));
        
        const Eigen::Vector3<T> camera_coordinates = pose.matrix3x4() * map_coordinate;
        T d = camera_coordinates(2);
        T x = FOCAL_LENGTH * camera_coordinates(0) / d + X_CAMERA_OFFSET;
        T y = FOCAL_LENGTH * camera_coordinates(1) / d + Y_CAMERA_OFFSET;
        residuals[0] = observed_x - x;
        residuals[1] = observed_y - y;
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y, Eigen::Vector4<double> map_coordinate)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 6>(
            new BundleError(observed_x, observed_y, map_coordinate)));
    }

private:
    double observed_x;
    double observed_y;
    Eigen::Vector4<double> map_coordinate;
};

BundleAdjustment::BundleAdjustment(std::vector<MapPoint*> map_points, KeyFrame *frame)
{
    if (map_points.size() == 0) {
        return;
    }
    for (MapPoint *mp : map_points)
    {
        std::pair<float, float> camera_coordinates = frame->fromWorldToImage(mp->wcoord);
        float u = camera_coordinates.first;
        float v = camera_coordinates.second;
        int min_hamming_distance = 10000;
        int current_hamming_distance = -1;
        cv::KeyPoint right_kp;
        for (int i = 0; i < frame->keypoints.size(); i++)
        {
            if (frame->keypoints[i].pt.x - WINDOW > u || frame->keypoints[i].pt.x + WINDOW < u)
                continue;
            if (frame->keypoints[i].pt.y - WINDOW > v || frame->keypoints[i].pt.y + WINDOW < v)
                continue;
            current_hamming_distance = ComputeHammingDistance(mp->orb_descriptor, frame->orb_descriptors.row(i));
            if (current_hamming_distance < min_hamming_distance)
            {
                min_hamming_distance = current_hamming_distance;
                right_kp = frame->keypoints[i];
            }
        }
        if (current_hamming_distance == -1)
            continue;
        this->kps.push_back(right_kp);
        this->map_points.push_back(mp);
    }
}

Sophus::SE3d BundleAdjustment::solve()
{
    ceres::Problem problem;
    std::cout << this->kps.size() << " " <<  this->map_points.size() << "\n";
    for (int i = 0; i < this->map_points.size(); i++)
    {
        ceres::CostFunction *cost_function;
        cost_function = BundleError::Create(this->kps[i].pt.x, this->kps[i].pt.y, this->map_points[i]->wcoord);
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
        // ceres::LossFunction *loss_function = new ceres::CauchyLoss(0.5);
        problem.AddResidualBlock(cost_function, loss_function, this->T.data());
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    return this->T;
    
}