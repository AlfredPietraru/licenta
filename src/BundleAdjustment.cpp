#include "../include/BundleAdjustment.h"
#include "../include/rotation.h"


// de rezolvat corespunzator bundle adjustment, de adaugat
class BundleError
{
    // const double FOCAL_LENGTH = 1436.71;
    // const double X_CAMERA_OFFSET = 757.49;
    // const double Y_CAMERA_OFFSET = 1017.88;
    const double BASELINE = 0.08;

public:
    BundleError(Eigen::Vector3d observed, Eigen::Vector4<double> map_coordinate, double scale_sigma, Eigen::Matrix3d K, 
        bool is_monocular) : observed(observed),
                 map_coordinate(map_coordinate), scale_sigma(scale_sigma), K(K), is_monocular(is_monocular) {}

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
        T x = K(0, 0) * camera_coordinates(0) / d + K(0, 2);
        T y = K(1, 1) * camera_coordinates(1) / d + K(1, 2);
        residuals[0] = (x - observed(0)) / scale_sigma;
        residuals[1] = (y - observed(1)) / scale_sigma;

        if (this->is_monocular) return true;
        T z_projected = K(0, 0) * (camera_coordinates(0) - BASELINE) / d + K(0, 2);
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
    bool is_monocular;
    Eigen::Matrix3d K;
    Eigen::Vector3d observed;
    Eigen::Vector4d map_coordinate;
    double scale_sigma;
};

Sophus::SE3d BundleAdjustment::solve(KeyFrame *kf, std::vector<MapPoint*> map_points, std::vector<cv::KeyPoint> kps)
{ 
    const double BASELINE = 0.08;
    if (map_points.size() == 0) return kf->Tiw;
    double *data = kf->Tiw.data();
    ceres::Problem problem;
    int nr_monocular_points = 0;
    int nr_stereo_points = 0;
    for (int i = 0; i < map_points.size(); i++)
    {
        ceres::CostFunction *cost_function;
        double sigma = std::pow(1.2, kps[i].octave);
        std::cout << sigma << " ";
        float dd = kf->compute_depth_in_keypoint(kps[i]);
        if (dd <= 0) {
            nr_monocular_points ++;
            cost_function = BundleError::Create_Monocular(Eigen::Vector3d(kps[i].pt.x, kps[i].pt.y, 0), map_points[i]->wcoord, sigma, kf->K);
        } else {
            nr_stereo_points++;
            double fake_right_coordinate = kps[i].pt.x - (kf->K(0, 0) * BASELINE / dd);
            cost_function = BundleError::Create_Stereo(Eigen::Vector3d(kps[i].pt.x, kps[i].pt.y, fake_right_coordinate), map_points[i]->wcoord, sigma, kf->K);
        }
        ceres::LossFunction *loss_function = new ceres::HuberLoss(HUBER_LOSS_VALUE);
        problem.AddResidualBlock(cost_function, loss_function, data);
    }
    std::cout << "au fost gasite atatea puncte monoculare " << nr_monocular_points << "\n";
    std::cout << "au fost gasite atatea puncte stereo " << nr_stereo_points << "\n";
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
    return Sophus::SE3d(q, translation);
    
}