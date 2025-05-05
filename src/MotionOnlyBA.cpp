#include "../include/MotionOnlyBA.h"
using SE3Manifold = ceres::ProductManifold<ceres::EigenQuaternionManifold, ceres::EuclideanManifold<3>>;

class BundleError
{
public:
    BundleError(KeyFrame *kf, MapPoint *mp, Feature *f, bool is_monocular) : kf(kf), mp(mp), f(f), is_monocular(is_monocular) {}

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        T map_coordinate_world[3] = {T(mp->wcoord_3d(0)), T(mp->wcoord_3d(1)), T(mp->wcoord_3d(2))};
        T camera_coordinates[3];
        ceres::QuaternionRotatePoint(pose, map_coordinate_world, camera_coordinates);
        camera_coordinates[0] += pose[4];
        camera_coordinates[1] += pose[5];
        camera_coordinates[2] += pose[6];
        camera_coordinates[2] = ceres::fmax(camera_coordinates[2], 1e-1);
        T inv_d = T(1) / camera_coordinates[2];
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        residuals[0] = (x - T(f->kpu.pt.x)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) / kf->POW_OCTAVE[f->kpu.octave];
        if (this->is_monocular)
            return true;

        T disp_pred = T(kf->K(0, 0)) * 0.08 * inv_d;
        T disp_meas = T(f->kpu.pt.x) - T(f->right_coordinate);      
        residuals[2] = (disp_pred - disp_meas) / kf->POW_OCTAVE[f->kpu.octave];
        return true;
    }

    static ceres::CostFunction *Create_Monocular(KeyFrame *kf, MapPoint *mp, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 7>(
            new BundleError(kf, mp, f, true)));
    }

    static ceres::CostFunction *Create_Stereo(KeyFrame *kf, MapPoint *mp, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 3, 7>(
            new BundleError(kf, mp, f, false)));
    }

private:
    KeyFrame *kf;
    MapPoint* mp;
    Feature *f;
    bool is_monocular;
};

Sophus::SE3d compute_pose(KeyFrame *kf, double *pose) {
    Eigen::Quaterniond quaternion(pose[0], pose[1], pose[2], pose[3]);
    Eigen::Quaterniond old_quaternion = kf->Tcw.unit_quaternion();
    if (old_quaternion.dot(quaternion) < 0)
    {
        quaternion.coeffs() *= -1;
    }
    return Sophus::SE3d(quaternion, Eigen::Vector3d(pose[4], pose[5], pose[6]));
}


void MotionOnlyBA::solve_ceres(KeyFrame *kf, bool display)
{
    if (kf->mp_correlations.size() < 3)
        return ;
    
    const float chi2Mono[4]={5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4]={7.815, 7.815, 7.815, 7.815};
    double error;
    for (int i = 0; i < 4; i++)
    {
        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
        options.function_tolerance = 1e-8;
        options.gradient_tolerance = 1e-7;
        options.parameter_tolerance = 1e-7;

        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        options.check_gradients = false;
        options.minimizer_progress_to_stdout = display;
        options.max_num_iterations = 10;
        
        ceres::LossFunction *loss_function_mono = new ceres::HuberLoss(sqrt(chi2Mono[i]));
        ceres::LossFunction *loss_function_stereo = new ceres::HuberLoss(sqrt(chi2Stereo[i]));

        problem.AddParameterBlock(kf->pose_vector, 7);
        problem.SetManifold(kf->pose_vector, new SE3Manifold());
        for (std::pair<MapPoint*, Feature*> it : kf->mp_correlations) {
            MapPoint *mp = it.first;
            Feature *f = it.second;
            if (f->is_monocular) {
                error = Common::get_monocular_reprojection_error(kf, mp, f, chi2Mono[i]);
                if (error > chi2Mono[i]) continue;
                ceres::CostFunction *cost_function;
                cost_function = BundleError::Create_Monocular(kf, mp, f);
                problem.AddResidualBlock(cost_function, loss_function_mono, kf->pose_vector);   
                continue; 
            }
            if (!f->is_monocular) {
                error = Common::get_rgbd_reprojection_error(kf, mp, f, chi2Stereo[i]);
                if (error > chi2Stereo[i]) continue;
                ceres::CostFunction *cost_function;
                cost_function = BundleError::Create_Stereo(kf, mp, f);    
                problem.AddResidualBlock(cost_function, loss_function_stereo, kf->pose_vector);
            }
        }

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (display) {
            std::cout << summary.FullReport() << "\n";
        }
        kf->set_keyframe_position(kf->compute_pose()); 
    }
}