#ifndef BUNDLE_ADJUSTMENT_H
#define BUNDLE_ADJUSTMENT_H

#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include "ceres/ceres.h"
#include <sophus/se3.hpp>
#include "MapPoint.h"
#include "KeyFrame.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include <iostream>

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

class MotionOnlyBA
{
public:     
    MotionOnlyBA() {};
    Sophus::SE3d solve_ceres(KeyFrame *frame, bool display);
    Sophus::SE3d solve_g2o(KeyFrame *frame);
    double get_rgbd_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* feature, double chi2);
    double get_monocular_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* feature, double chi2);
};

#endif
