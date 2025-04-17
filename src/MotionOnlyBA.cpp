#include "../include/MotionOnlyBA.h"
#include "../include/rotation.h"

class BundleError
{
public:
    BundleError(Eigen::Vector3d observed, Eigen::Vector4<double> map_coordinate, double scale_sigma, Eigen::Matrix3d K,
                bool is_monocular) : observed(observed),
                                     map_coordinate(map_coordinate), scale_sigma(scale_sigma), K(K), is_monocular(is_monocular) {}

    template <typename T>
    bool operator()(const T *const pose, T *residuals) const
    {
        T map_coordinate_world[3] = {T(map_coordinate(0)), T(map_coordinate(1)), T(map_coordinate(2))};
        T camera_coordinates[3];
        T quat[4];
        quat[0] = pose[0];
        quat[1] = pose[1];
        quat[2] = pose[2];
        quat[3] = pose[3];
        ceres::QuaternionRotatePoint(quat, map_coordinate_world, camera_coordinates);
        camera_coordinates[0] += pose[4];
        camera_coordinates[1] += pose[5];
        camera_coordinates[2] += pose[6];
        T inv_d = T(1) / (camera_coordinates[2] + T(1e-6));
        T x = T(K(0, 0)) * camera_coordinates[0] * inv_d + T(K(0, 2));
        T y = T(K(1, 1)) * camera_coordinates[1] * inv_d + T(K(1, 2));
        residuals[0] = (x - observed(0)) / scale_sigma;
        residuals[1] = (y - observed(1)) / scale_sigma;
        if (this->is_monocular)
            return true;

        T z_projected = x - T(K(0, 0)) * BASELINE * inv_d;
        residuals[2] = (z_projected - observed(2)) / scale_sigma;
        return true;
    }

    static ceres::CostFunction *Create_Monocular(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 2, 7>(
            new BundleError(observed, map_coordinate, scale_sigma, K, true)));
    }

    static ceres::CostFunction *Create_Stereo(Eigen::Vector3d observed, Eigen::Vector4d map_coordinate, double scale_sigma, Eigen::Matrix3d K)
    {
        return (new ceres::AutoDiffCostFunction<BundleError, 3, 7>(
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

double get_rgbd_reprojection_error(MapPoint *mp, Eigen::Matrix3d K, Sophus::SE3d pose, Feature* feature, double chi2) {
    double residuals[3];
    Eigen::Vector3d camera_coordinates = pose.matrix3x4() * mp->wcoord;
    if (camera_coordinates[2] <= 1e-6) return 100000;
    double inv_d = 1 / camera_coordinates[2];
    double x = K(0, 0) * camera_coordinates[0] * inv_d + K(0, 2);
    double y = K(1, 1) * camera_coordinates[1] * inv_d + K(1, 2);
    cv::KeyPoint kpu = feature->kpu;
    residuals[0] = (x - kpu.pt.x) / std::pow(1.2, kpu.octave);
    residuals[1] = (y - kpu.pt.y) / std::pow(1.2, kpu.octave);
    double z_projected = x - K(0, 0) * 0.08 * inv_d;
    residuals[2] = (z_projected - feature->stereo_depth) / std::pow(1.2, kpu.octave);
    double a = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]);
    if (a <= sqrt(chi2)) return pow(a, 2) / 2;
    return sqrt(chi2) * a - chi2 / 2;
}


double get_monocular_reprojection_error(MapPoint *mp, Eigen::Matrix3d K, Sophus::SE3d pose, Feature* feature, double chi2) {
    double residuals[2];
    Eigen::Vector3d camera_coordinates = pose.matrix3x4() * mp->wcoord;
    if (camera_coordinates[2] <= 1e-6) return 100000;
    double inv_d = 1 / camera_coordinates[2];
    double x = K(0, 0) * camera_coordinates[0] * inv_d + K(0, 2);
    double y = K(1, 1) * camera_coordinates[1] * inv_d + K(1, 2);
    cv::KeyPoint kpu = feature->kpu;
    residuals[0] = (x - kpu.pt.x) / std::pow(1.2, kpu.octave);
    residuals[1] = (y - kpu.pt.y) / std::pow(1.2, kpu.octave);
    double a = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1]);
    if (a <= sqrt(chi2)) return pow(a, 2) / 2;
    return sqrt(chi2) * a - chi2 / 2;
}

Sophus::SE3d compute_pose(KeyFrame *kf, double *pose) {
    Eigen::Quaterniond quaternion(pose[0], pose[1], pose[2], pose[3]);
    Eigen::Quaterniond old_quaternion = kf->Tiw.unit_quaternion();
    if (old_quaternion.dot(quaternion) < 0)
    {
        quaternion.coeffs() *= -1;
    }
    return Sophus::SE3d(quaternion, Eigen::Vector3d(pose[4], pose[5], pose[6]));
}

Sophus::SE3d BundleAdjustment::solve_ceres(KeyFrame *kf, std::unordered_map<MapPoint *, Feature *> &matches)
{
    if (matches.size() < 3)
        return kf->Tiw;
    
    const float chi2Mono[4]={5.991, 5.991, 5.991, 5.991};
    const float chi2Stereo[4]={7.815, 7.815, 7.815, 7.815};
    Sophus::SE3d pose = kf->Tiw;
    const double BASELINE = 0.08;
    Eigen::Quaterniond quat = pose.unit_quaternion();
    double pose_vector[7];
    pose_vector[0] = quat.w();
    pose_vector[1] = quat.x();
    pose_vector[2] = quat.y();
    pose_vector[3] = quat.z();
    pose_vector[4] = pose.translation().x();
    pose_vector[5] = pose.translation().y();
    pose_vector[6] = pose.translation().z();
    std::unordered_map<MapPoint *, Feature *> mono_matches;
    std::unordered_map<MapPoint *, Feature *> rgbd_matches;
    for (auto it = matches.begin(); it != matches.end(); it++) {
        if (it->second->stereo_depth <= 1e-6)
        {
            mono_matches.insert({it->first, it->second});
            continue;
        }
        rgbd_matches.insert({it->first, it->second});
    }


    for (int i = 0; i < 4; i++)
    {

        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::LinearSolverType::DENSE_QR;
        options.function_tolerance = 1e-7;
        options.gradient_tolerance = 1e-7;
        options.parameter_tolerance = 1e-8;

        options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
        // options.check_gradients = true;
        // options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 10;
        
        ceres::LossFunction *loss_function_mono = new ceres::HuberLoss(sqrt(chi2Mono[i]));
        ceres::LossFunction *loss_function_stereo = new ceres::HuberLoss(sqrt(chi2Stereo[i]));
        
        int inlier = 0;
        for (auto it = mono_matches.begin(); it != mono_matches.end(); it++) {
            MapPoint *mp = it->first;
            if (kf->check_map_point_outlier(mp))  continue;
            ceres::CostFunction *cost_function;
            cv::KeyPoint kpu = it->second->kpu;
            cost_function = BundleError::Create_Monocular(Eigen::Vector3d(kpu.pt.x, kpu.pt.y, 0), mp->wcoord,
                                                              std::pow(1.2, kpu.octave), kf->K);
            problem.AddResidualBlock(cost_function, loss_function_mono, pose_vector);
        }

        for (auto it = rgbd_matches.begin(); it != rgbd_matches.end(); it++) {
            MapPoint *mp = it->first;
            if (kf->check_map_point_outlier(mp))  continue;
            ceres::CostFunction *cost_function;
            cv::KeyPoint kpu = it->second->kpu;
            cost_function = BundleError::Create_Stereo(Eigen::Vector3d(kpu.pt.x, kpu.pt.y, it->second->stereo_depth), mp->wcoord,
                            std::pow(1.2, kpu.octave), kf->K);    
            ceres::ResidualBlockId id = problem.AddResidualBlock(cost_function, loss_function_stereo, pose_vector);
        }

        using SE3Manifold = ceres::ProductManifold<ceres::QuaternionManifold, ceres::EuclideanManifold<3>>;
        problem.AddParameterBlock(pose_vector, 7);
        problem.SetManifold(pose_vector, new SE3Manifold());

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        // std::cout << summary.FullReport() << "\n";

        Sophus::SE3d intermediate_pose = compute_pose(kf, pose_vector);
        double error;
        
        for (auto it = mono_matches.begin(); it != mono_matches.end(); it++) {
            MapPoint *mp = it->first;
            error = get_monocular_reprojection_error(mp, kf->K, intermediate_pose, it->second, chi2Mono[i]);  
            if (error > chi2Mono[i]) {
                kf->add_outlier_element(mp);
            } else {
                kf->remove_outlier_element(mp);
                inlier++;
            }
        }

        for (auto it = rgbd_matches.begin(); it != rgbd_matches.end(); it++) {
            MapPoint *mp = it->first;
            error = get_rgbd_reprojection_error(mp, kf->K, intermediate_pose, it->second, chi2Stereo[i]);
            if (error > chi2Stereo[i]) {
                kf->add_outlier_element(mp);
            } else {
                kf->remove_outlier_element(mp);
                inlier++;
            }
        }
    }
    return compute_pose(kf, pose_vector);    
}

Sophus::SE3d BundleAdjustment::solve_g2o(KeyFrame *kf, std::unordered_map<MapPoint *, Feature *> &matches)
{

    std::unique_ptr<LinearSolverType> linearSolver(new LinearSolverType());
    // Create the block solver by moving the linear solver into it.
    std::unique_ptr<BlockSolverType> blockSolver(new BlockSolverType(std::move(linearSolver)));
    // Create the OptimizationAlgorithmLevenberg by moving the block solver into it.
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // Set Frame vertex
    g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    Sophus::SE3d pose = kf->Tiw;
    vSE3->setEstimate(g2o::SE3Quat(pose.rotationMatrix(), pose.translation()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    std::vector<std::pair<MapPoint *, Feature *>> matches_vector(matches.begin(), matches.end());
    const int N = matches_vector.size();

    std::vector<g2o::EdgeSE3ProjectXYZOnlyPose*> vpEdgesMono;
    std::vector<size_t> vnIndexEdgeMono;
    vpEdgesMono.reserve(N);
    vnIndexEdgeMono.reserve(N);

    std::vector<g2o::EdgeStereoSE3ProjectXYZOnlyPose*> vpEdgesStereo;
    std::vector<size_t> vnIndexEdgeStereo;
    vpEdgesStereo.reserve(N);
    vnIndexEdgeStereo.reserve(N);

    const float deltaMono = sqrt(5.991);
    const float deltaStereo = sqrt(7.815);

    for(int i = 0; i < matches_vector.size(); i++) 
    {
        MapPoint* pMP = matches_vector[i].first;
        Feature *f = matches_vector[i].second;
        if(pMP)
        {
            // Monocular observation
            if(f->depth <= 1e-6)
            {

                Eigen::Matrix<double,2,1> obs;
                const cv::KeyPoint &kpUn = f->kpu;
                g2o::EdgeSE3ProjectXYZOnlyPose* e = new g2o::EdgeSE3ProjectXYZOnlyPose();
                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(g2o::Vector2(kpUn.pt.x, kpUn.pt.y));
                const float invSigma2 = 1 / pow(1.2, 2* kpUn.octave);
                e->setInformation(Eigen::Matrix2d::Identity()*invSigma2);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaMono);

                e->fx = kf->K(0, 0);
                e->fy = kf->K(1, 1);
                e->cx = kf->K(0, 2);
                e->cy = kf->K(1, 2);
                e->Xw[0] = pMP->wcoord_3d(0);
                e->Xw[1] = pMP->wcoord_3d(1);
                e->Xw[2] = pMP->wcoord_3d(2);
                optimizer.addEdge(e);

                vpEdgesMono.push_back(e);
                vnIndexEdgeMono.push_back(i);
            }
            else 
            {

                const cv::KeyPoint &kpUn = f->kpu;
                g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();

                e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
                e->setMeasurement(g2o::Vector3(kpUn.pt.x, kpUn.pt.y, f->stereo_depth));
                const float invSigma2 = 1 / pow(1.2, 2* kpUn.octave);
                Eigen::Matrix3d Info = Eigen::Matrix3d::Identity()*invSigma2;
                e->setInformation(Info);

                g2o::RobustKernelHuber* rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                rk->setDelta(deltaStereo);

                e->fx = kf->K(0, 0);
                e->fy = kf->K(1, 1);
                e->cx = kf->K(0, 2);
                e->cy = kf->K(1, 2);
                e->bf = 41.38451264;
                e->Xw[0] = pMP->wcoord_3d(0);
                e->Xw[1] = pMP->wcoord_3d(1);
                e->Xw[2] = pMP->wcoord_3d(2);

                optimizer.addEdge(e);

                vpEdgesStereo.push_back(e);
                vnIndexEdgeStereo.push_back(i);
            }
        }
    }

    const float chi2Mono[4]={5.991,5.991,5.991,5.991};
    const float chi2Stereo[4]={7.815,7.815,7.815, 7.815};
    const int its[4]={10,10,10,10};    

    for(size_t it=0; it<4; it++)
    {
        vSE3->setEstimate(g2o::SE3Quat(pose.rotationMatrix(), pose.translation()));
        optimizer.initializeOptimization(0);
        optimizer.optimize(its[it]);

        for(size_t i=0, iend=vpEdgesMono.size(); i<iend; i++)
        {
            g2o::EdgeSE3ProjectXYZOnlyPose* e = vpEdgesMono[i];

            const size_t idx = vnIndexEdgeMono[i];
            MapPoint *mp = matches_vector[idx].first;
            if(kf->check_map_point_outlier(mp))
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Mono[it])
            {   
                kf->add_outlier_element(mp);             
                e->setLevel(1);
            }
            else
            {
                kf->remove_outlier_element(mp);
                e->setLevel(0);
            }
            if(it==2)
                e->setRobustKernel(0);
        }

        for(size_t i=0, iend=vpEdgesStereo.size(); i<iend; i++)
        {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose* e = vpEdgesStereo[i];

            const size_t idx = vnIndexEdgeStereo[i];
            MapPoint *mp = matches_vector[idx].first;
            if(kf->check_map_point_outlier(mp))
            {
                e->computeError();
            }

            const float chi2 = e->chi2();

            if(chi2>chi2Stereo[it])
            {
                kf->add_outlier_element(mp);
                e->setLevel(1);
            }
            else
            {                
                kf->remove_outlier_element(mp);
                e->setLevel(0);
            }

            if(it==2)
                e->setRobustKernel(0);
        }
        if(optimizer.edges().size()<10)
            break;
        g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
        g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
        pose = Sophus::SE3d(SE3quat_recov.to_homogeneous_matrix()); 
    }    

    g2o::VertexSE3Expmap* vSE3_recov = static_cast<g2o::VertexSE3Expmap*>(optimizer.vertex(0));
    g2o::SE3Quat SE3quat_recov = vSE3_recov->estimate();
    return Sophus::SE3d(SE3quat_recov.to_homogeneous_matrix());
}