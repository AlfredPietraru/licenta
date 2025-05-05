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

Sophus::SE3d MotionOnlyBA::solve_g2o(KeyFrame *kf)
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
    Sophus::SE3d pose = kf->Tcw;
    vSE3->setEstimate(g2o::SE3Quat(pose.rotationMatrix(), pose.translation()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    // Set MapPoint vertices
    std::vector<std::pair<MapPoint *, Feature *>> matches_vector(kf->mp_correlations.begin(), kf->mp_correlations.end());
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

    for(long unsigned int i = 0; i < matches_vector.size(); i++) 
    {
        MapPoint* pMP = matches_vector[i].first;
        Feature *f = matches_vector[i].second;
        if(pMP)
        {
            // Monocular observation
            if(f->is_monocular)
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
                e->setMeasurement(g2o::Vector3(kpUn.pt.x, kpUn.pt.y, f->right_coordinate));
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