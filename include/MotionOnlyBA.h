#ifndef BUNDLE_ADJUSTMENT_H
#define BUNDLE_ADJUSTMENT_H

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
#include "../include/rotation.h"
#include <iostream>
#include "Common.h"

typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType;
typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;

class MotionOnlyBA
{
public:     
    MotionOnlyBA() {};
    void solve_ceres(KeyFrame *frame, bool display);
    void RemoveOutliersCurrentFrame(KeyFrame *kf, double chi2Mono, double chi2Stereo);
};

#endif
