#include "../include/BundleAdjustment.h"


Sophus::SE3d BundleAdjustment::solve_ceres(Map *mapp, KeyFrame *frame) {
   std::unordered_set<KeyFrame*> local_keyframes = mapp->get_local_keyframes(frame);
    if (local_keyframes.size() == 0) {
        std::cout << "CEVA E GRESIT IN BUNDLE ADJUSTMENT ACEST KEYFRAME ESTE IZOLAT\n";
        return frame->Tcw;
    }
    std::unordered_set<MapPoint*> local_map_points;
    for (KeyFrame *kf : local_keyframes) {
        local_map_points.insert(kf->map_points.begin(), kf->map_points.end());
    }
    if (local_map_points.size() == 0) {
        std::cout << "CEVA NU E BINE NU EXISTA DELOC PUNCTE IN LOCAL MAP PENTRU OPTIMIZARE\n";
    }
    std::unordered_set<KeyFrame*> fixed_keyframes;
    for (MapPoint *mp : local_map_points) {
        for (KeyFrame *kf_which_sees_map_point : mp->keyframes) {
            if (local_keyframes.find(kf_which_sees_map_point) == local_keyframes.end()) fixed_keyframes.insert(kf_which_sees_map_point);
        } 
    }
    // ceres::Problem problem;
    // ceres::Solver::Options options;
    // options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
    // options.function_tolerance = 1e-7;
    // options.gradient_tolerance = 1e-7;
    // options.parameter_tolerance = 1e-8;

    // options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    // options.max_num_iterations = 5;
    // double chi2Mono = 5.991;
    // double chi2Stereo = 7.815;
    // ceres::LossFunction *loss_function_mono = new ceres::HuberLoss(sqrt(chi2Mono));
    // ceres::LossFunction *loss_function_stereo = new ceres::HuberLoss(sqrt(chi2Stereo));    
    return frame->Tcw;
}