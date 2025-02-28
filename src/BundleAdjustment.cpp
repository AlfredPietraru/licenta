#include "../include/BundleAdjustment.h"

class BundleError
{
    const double FOCAL_LENGTH = 500.0;
    const double X_CAMERA_OFFSET = 325.1;
    const double Y_CAMERA_OFFSET = 250.7;

public:
    BundleError(double observation_x, double observation_y, Eigen::Vector4<double> map_coordinate) : observed_x(observation_x),
                                                              observed_y(observation_y), map_coordinate(map_coordinate) {}

    template <typename T>
    bool operator()(const T *const rotation, const T *const translation, T *residuals) const
    {
        Eigen::Matrix4<T> eigen_pose;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                eigen_pose(i, j) = rotation[i * 3 + j];
            }
        }
        for (int i = 0; i < 3; i++)
        {
            eigen_pose(i, 3) = translation[i];
        }
        for (int i = 0; i < 2; i++)
        {
            eigen_pose(3, i) = (T)0;
        }
        eigen_pose(3, 3) = (T)1;

        const Eigen::Vector4<T> camera_coordinates = eigen_pose * map_coordinate;
        T d = camera_coordinates(2);
        T x = FOCAL_LENGTH * camera_coordinates(0) / d + X_CAMERA_OFFSET;
        T y = FOCAL_LENGTH * camera_coordinates(1) / d + Y_CAMERA_OFFSET;
        residuals[0] = observed_x - x;
        residuals[1] = observed_y - y;
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y, Eigen::Vector4<double> map_coordinate)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 9, 3>(
            new BundleError(observed_x, observed_y, map_coordinate)));
    }

private:
    double observed_x;
    double observed_y;
    Eigen::Vector4<double> map_coordinate;
};

BundleAdjustment::BundleAdjustment(std::vector<MapPoint*> map_points, KeyFrame *frame)
{
    this->map_points = map_points;
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
    }

    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            this->rotation[i * 3 + j] = frame->Tiw(i, j);
        }
    }
    for (int i = 0; i < 3; i++) {
        this->translation[i] = frame->Tiw(i, 3);   
    }
}

Eigen::Matrix4d BundleAdjustment::return_optimized_pose()
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    // Eigen::Matrix3d R;
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 3; j++)
        {
            T(i, j) = this->rotation[i * 3 + j];
        }
    }
    // R = R * ((R.transpose() * R).cwiseSqrt()).inverse();
    // for (int i = 0; i < 3; i++) {
    //     for (int j = 0; j < 3; j++) {
    //         T(i, j) = R(i, j);
    //     }
    // }
    for (int i = 0; i < 3; i++)
    {
        T(i, 3) = this->translation[9 + i];
    }

    return T;
}

bool BundleAdjustment::enough_points_tracker()
{
    return this->map_points.size() > 50;
}

void BundleAdjustment::solve()
{
    ceres::Problem problem;

    for (int i = 0; i < this->map_points.size(); i++)
    {
        ceres::CostFunction *cost_function;
        cost_function = BundleError::Create(kps[i].pt.x, kps[i].pt.y, this->map_points[i]->wcoord);
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
        problem.AddResidualBlock(cost_function, loss_function, this->rotation, this->translation);
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    // std::cout << summary.FullReport() << "\n";
}