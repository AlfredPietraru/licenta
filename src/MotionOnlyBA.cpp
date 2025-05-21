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
        residuals[0] = (x - T(f->kpu.pt.x)) * kf->INVERSE_POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) * kf->INVERSE_POW_OCTAVE[f->kpu.octave];
        if (this->is_monocular)
            return true;

        T disp_pred = T(kf->K(0, 0)) * kf->BASELINE * inv_d;
        T disp_meas = T(f->kpu.pt.x) - T(f->right_coordinate);      
        residuals[2] = (disp_pred - disp_meas) * kf->INVERSE_POW_OCTAVE[f->kpu.octave];
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


class MotionChangeError {
    public:
    MotionChangeError(KeyFrame *prev_kf, double weight) : prev_kf(prev_kf), weight(weight) {}

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        residuals[0] = (prev_kf->pose_vector[4] - pose[4]) * weight;
        residuals[1] = (prev_kf->pose_vector[5] - pose[5]) * weight;
        residuals[2] = (prev_kf->pose_vector[6] - pose[6]) * weight;

        T q_prev[4] = {T(prev_kf->pose_vector[0]), T(prev_kf->pose_vector[1]),
            T(prev_kf->pose_vector[2]), T(prev_kf->pose_vector[3])};
        const T* q_curr = pose;
        T q_prev_conj[4] = { q_prev[0], -q_prev[1], -q_prev[2], -q_prev[3] };
        T q_rel[4];
        ceres::QuaternionProduct(q_prev_conj, q_curr, q_rel);
        residuals[3] = T(2.0) * q_rel[1] * weight;
        residuals[4] = T(2.0) * q_rel[2] * weight;
        residuals[5] = T(2.0) * q_rel[3] * weight;
        return true;
    }

    static ceres::CostFunction *Create(KeyFrame *prev_kf, double weight)
    {
        return (new ceres::AutoDiffCostFunction<MotionChangeError, 6, 7>(
            new MotionChangeError(prev_kf, weight)));
    }

    private:
    KeyFrame *prev_kf;
    double weight;
};

void MotionOnlyBA::RemoveOutliersCurrentFrame(KeyFrame *kf, double chi2Mono, double chi2Stereo) {
    double error;
    bool should_delete;
    std::vector<MapPoint*> to_delete;
    to_delete.reserve(kf->mp_correlations.size());
    for (std::pair<MapPoint*, Feature*> it : kf->mp_correlations) {
        MapPoint *mp = it.first;
        Feature *f = it.second;
        if (f->is_monocular) {
            error = Common::get_monocular_reprojection_error(kf, mp, f, chi2Mono);
            should_delete = error > chi2Mono; 
        } else {
            error = Common::get_rgbd_reprojection_error(kf, mp, f, chi2Stereo);
            should_delete = error > chi2Stereo;
        }
        if (!should_delete) continue;
        to_delete.push_back(mp);
    }

    for (MapPoint *mp : to_delete) {
        Map::remove_map_point_from_keyframe(kf, mp);
    }

}

void MotionOnlyBA::solve_ceres(KeyFrame *kf, KeyFrame *prev_kf, bool display)
{
    if (kf->mp_correlations.size() < 3)
        return ;

    for (int i = 0; i < 1; i++)
    {
        ceres::Problem problem;
        options.check_gradients = false;
        options.minimizer_progress_to_stdout = display;        
        ceres::LossFunction *loss_function_mono = new ceres::HuberLoss(sqrt(chi2Mono[i]));
        ceres::LossFunction *loss_function_stereo = new ceres::HuberLoss(sqrt(chi2Stereo[i]));

        problem.AddParameterBlock(kf->pose_vector, 7);
        problem.SetManifold(kf->pose_vector, new SE3Manifold());
        if (prev_kf != nullptr) {
            ceres::CostFunction *cost_function;
            cost_function = MotionChangeError::Create(prev_kf, 100);
            problem.AddResidualBlock(cost_function, nullptr, kf->pose_vector);
        }

        for (std::pair<MapPoint*, Feature*> it : kf->mp_correlations) {
            MapPoint *mp = it.first;
            Feature *f = it.second;
            if (f->is_monocular) {
                ceres::CostFunction *cost_function;
                cost_function = BundleError::Create_Monocular(kf, mp, f);
                problem.AddResidualBlock(cost_function, loss_function_mono, kf->pose_vector);   
                continue; 
            }
            if (!f->is_monocular) {
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
        if (i == 0) this->RemoveOutliersCurrentFrame(kf, chi2Mono[i], chi2Stereo[i]);
        if (kf->mp_correlations.size() < 3) return;
    }
}