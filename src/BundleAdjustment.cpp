#include "../include/BundleAdjustment.h"
#include "../include/rotation.h"


// de rezolvat corespunzator bundle adjustment, de adaugat
class BundleError
{
    const double FOCAL_LENGTH = 132.28;
    const double X_CAMERA_OFFSET = 110.1;
    const double Y_CAMERA_OFFSET = 115.7;
    const double BASELINE = 0.08;

public:
    BundleError(Eigen::Vector3d observed, Eigen::Vector4<double> map_coordinate, double scale_sigma) : observed(observed),
                 map_coordinate(map_coordinate), scale_sigma(scale_sigma) {}

    template <typename T>
    bool operator()(const T *const se3, T *residuals) const
    {
        Eigen::Vector3<T> translation;
        for (int i = 0; i < 3; i++) {
            translation(i) = se3[i + 4];
        }
        // de lucrat la bundle adjustment
        // problema cu quaternionii, imi dadeau cu minus unele elemente de pe diagonala principala 
        Eigen::Quaternion<T> q(se3[3], se3[0], se3[1], se3[2]); 
        q.normalize();
        Sophus::SE3<T> pose = Sophus::SE3<T>(q, translation);
        
        const Eigen::Vector3<T> camera_coordinates = pose.matrix3x4() * map_coordinate;
        T d = camera_coordinates(2);
        // if (d < 0) return false;
        Eigen::Vector3<T> val;
        T x = FOCAL_LENGTH * camera_coordinates(0) / d + X_CAMERA_OFFSET;
        T y = FOCAL_LENGTH * camera_coordinates(1) / d + Y_CAMERA_OFFSET;
        val(0) = (x - observed(0)) / scale_sigma;
        val(1) = (y - observed(1)) / scale_sigma;
        if (observed(2) > -1e-7 && observed(2) < 1e-7) {
            val(2) = (T)0;
        } else {
            T z = (camera_coordinates(0) - BASELINE) / d + X_CAMERA_OFFSET;
            val(2) = (z - observed(2)) / scale_sigma;  
        }
        residuals[0] = val.norm(); 
        return true;
    }

    static ceres::CostFunction *Create(Eigen::Vector3d observed, Eigen::Vector4<double> map_coordinate, double scale_sigma)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 1, 6>(
            new BundleError(observed, map_coordinate, scale_sigma)));
    }

private:
    Eigen::Vector3<double> observed;
    Eigen::Vector4<double> map_coordinate;
    double scale_sigma;
};

Sophus::SE3d BundleAdjustment::solve(KeyFrame *kf, std::vector<MapPoint*> map_points, std::vector<cv::KeyPoint> kps)
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

    double *data = kf->Tiw.data();
    ceres::Problem problem;
    if (map_points.size() == 0) return kf->Tiw;
    for (int i = 0; i < map_points.size(); i++)
    {
        ceres::CostFunction *cost_function;
        double sigma = std::pow(1.2, kps[i].octave);
        // std::cout << sigma << " ";
        float d = kf->depth_matrix.at<float>(kps[i].pt.x, kps[i].pt.y);
        if (d <= 0) {
            std::cout << "pe aici vreodata" << "\n\n\n";
            cost_function = BundleError::Create(Eigen::Vector3d(kps[i].pt.x, kps[i].pt.y, 0), map_points[i]->wcoord, sigma);
        } else {
            cost_function = BundleError::Create(Eigen::Vector3d(kps[i].pt.x, kps[i].pt.y, d), map_points[i]->wcoord, sigma);
        }
        ceres::LossFunction *loss_function = new ceres::HuberLoss(HUBER_LOSS_VALUE);
        problem.AddResidualBlock(cost_function, loss_function, data);
    }
    // std::cout << "\n";
    ceres::Solver::Options options;
    options.linear_solver_type = solver;
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
    Sophus::SE3d T = Sophus::SE3d(q, translation);
    return T;
    
}