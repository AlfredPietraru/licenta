#ifndef LOCAL_BUNDLE_ADJUSTMENT_H
#define LOCAL_BUNDLE_ADJUSTMENT_H
#include <iostream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <sophus/se3.hpp>
#include "ceres/ceres.h"
#include <iostream>
#include "rotation.h"
#include "Map.h"


class BundleAdjustmentVariableError
{
public:
    BundleAdjustmentVariableError(KeyFrame *kf, Feature *f, bool is_monocular) : kf(kf), f(f), is_monocular(is_monocular) {}

    template <typename T>
    bool operator()(const T *const pose, const T *map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        ceres::QuaternionRotatePoint(pose, map_coordinate, camera_coordinates);
        camera_coordinates[0] += pose[4];
        camera_coordinates[1] += pose[5];
        camera_coordinates[2] += pose[6];
        if (camera_coordinates[2] <= T(1e-3)) camera_coordinates[2] = T(1e-3);
        T inv_d = T(1) / camera_coordinates[2];
        
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        residuals[0] = (x - T(f->kpu.pt.x)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) / kf->POW_OCTAVE[f->kpu.octave];
        if (this->is_monocular) return true;

        T disp_pred = T(kf->K(0, 0)) * 0.08 * inv_d;
        T disp_meas = T(f->kpu.pt.x) - T(f->right_coordinate);      
        residuals[2] = (disp_pred - disp_meas) * kf->POW_OCTAVE[f->kpu.octave];
        return true;
    }

    static ceres::CostFunction *CreateMonocular(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentVariableError, 2, 7, 3>(new BundleAdjustmentVariableError(kf, f, true)));
    }

    static ceres::CostFunction *CreateStereo(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentVariableError, 3, 7, 3>(new BundleAdjustmentVariableError(kf, f, false)));
    }

    private:
    KeyFrame *kf;
    Feature *f;
    bool is_monocular;
};


class BundleAdjustmentStaticError
{
public:
    BundleAdjustmentStaticError(KeyFrame *kf, Feature *f, bool is_monocular) : kf(kf), f(f), is_monocular(is_monocular) {}

    template <typename T>
    bool operator()(const T *const map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        ceres::QuaternionRotatePoint((T*)kf->pose_vector, map_coordinate, camera_coordinates);
        camera_coordinates[0] += (T)kf->pose_vector[4];
        camera_coordinates[1] += (T)kf->pose_vector[5];
        camera_coordinates[2] += (T)kf->pose_vector[6];
        if (camera_coordinates[2] <= T(1e-3)) camera_coordinates[2] = T(1e-3); 
        T inv_d = T(1) / camera_coordinates[2];
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        residuals[0] = (x - T(f->kpu.pt.x)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) / kf->POW_OCTAVE[f->kpu.octave];
        if (this->is_monocular) return true;

        T disp_pred = T(kf->K(0, 0)) * 0.08 * inv_d;
        T disp_meas = T(f->kpu.pt.x) - T(f->right_coordinate);      
        residuals[2] = (disp_pred - disp_meas) * kf->POW_OCTAVE[f->kpu.octave];
        return true;
    }

    static ceres::CostFunction *CreateMonocular(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentStaticError, 2, 3>(new BundleAdjustmentStaticError(kf, f, true)));
    }

    static ceres::CostFunction *CreateStereo(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentStaticError, 3, 3>(new BundleAdjustmentStaticError(kf, f, false)));
    }

private:
    KeyFrame *kf;
    Feature *f;
    bool is_monocular;
};

class BundleAdjustment {
public:
    // const double POW_OCTAVE[10] = {1, 1.2, 1.44, 1.728, 2.0736, 2.48832, 2.985984, 3.5831808, 4.29981696, 5.159780352};
    BundleAdjustment() {};
    void solve_ceres(Map *mapp, KeyFrame *frame);
};

#endif