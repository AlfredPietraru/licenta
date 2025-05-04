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

class BundleAjustmentVariableKeyFrameMonocular {
public:
    BundleAjustmentVariableKeyFrameMonocular(KeyFrame *kf, Feature *f) : kf(kf), f(f) {}

    template <typename T>
    bool operator()(const T *const pose, const T *map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        ceres::QuaternionRotatePoint(pose, map_coordinate, camera_coordinates);
        camera_coordinates[0] += pose[4];
        camera_coordinates[1] += pose[5];
        camera_coordinates[2] += pose[6];
        camera_coordinates[2] = ceres::fmax(camera_coordinates[2], 1e-1);
        T inv_d = T(1) / camera_coordinates[2]; 
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        residuals[0] = (x - T(f->kpu.pt.x)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) / kf->POW_OCTAVE[f->kpu.octave];
        return true;
    }

    static ceres::CostFunction *CreateVariableMonocular(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAjustmentVariableKeyFrameMonocular, 2, 7, 3>(new BundleAjustmentVariableKeyFrameMonocular(kf, f)));
    }

private:
    KeyFrame *kf;
    Feature *f;
};

class BundleAjustmentVariableKeyFrameStereo {
public:
    BundleAjustmentVariableKeyFrameStereo(KeyFrame *kf, Feature *f) : kf(kf), f(f) {}
    template <typename T>
    bool operator()(const T *const pose, const T *map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        ceres::QuaternionRotatePoint(pose, map_coordinate, camera_coordinates);
        camera_coordinates[0] += pose[4];
        camera_coordinates[1] += pose[5];
        camera_coordinates[2] += pose[6];
        camera_coordinates[2] = ceres::fmax(camera_coordinates[2], 1e-1); 
        T inv_d = T(1) / camera_coordinates[2];
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        T disp_pred = T(kf->K(0, 0)) * kf->BASELINE * inv_d;
        T disp_meas = T(f->kpu.pt.x) - T(f->right_coordinate);
        residuals[0] = (x - T(f->kpu.pt.x)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[2] = (disp_pred - disp_meas) / kf->POW_OCTAVE[f->kpu.octave];
        return true;
    }

    static ceres::CostFunction *CreateVariableStereo(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAjustmentVariableKeyFrameStereo, 3, 7, 3>(new BundleAjustmentVariableKeyFrameStereo(kf, f)));
    }

private:
    KeyFrame *kf;
    Feature *f;
};

class BundleAjustmentFixedKeyFrameMonocular {
public:
    BundleAjustmentFixedKeyFrameMonocular(KeyFrame *kf, Feature *f) : kf(kf), f(f) {}
    template <typename T>
    bool operator()(const T *const map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        T pose[4];
        pose[0] = T(kf->pose_vector[0]);
        pose[1] = T(kf->pose_vector[1]);
        pose[2] = T(kf->pose_vector[2]);
        pose[3] = T(kf->pose_vector[3]);
        ceres::QuaternionRotatePoint(pose, map_coordinate, camera_coordinates);
        camera_coordinates[0] += (T)kf->pose_vector[4];
        camera_coordinates[1] += (T)kf->pose_vector[5];
        camera_coordinates[2] += (T)kf->pose_vector[6];
        camera_coordinates[2] = ceres::fmax(camera_coordinates[2], 1e-1);
        if (camera_coordinates[2] <= T(1e-1)) camera_coordinates[2] = T(1e-1); 
        T inv_d = T(1) / camera_coordinates[2];
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        residuals[0] = (x - T(f->kpu.pt.x)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) / kf->POW_OCTAVE[f->kpu.octave];
        return true;
    }

    static ceres::CostFunction *CreateFixedMonocular(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAjustmentFixedKeyFrameMonocular, 2, 3>(new BundleAjustmentFixedKeyFrameMonocular(kf, f)));
    }

private:
    KeyFrame *kf;
    Feature *f;
};

class BundleAjustmentFixedKeyFrameStereo {

public:
    BundleAjustmentFixedKeyFrameStereo(KeyFrame *kf, Feature *f) : kf(kf), f(f) {}

    template <typename T>
    bool operator()(const T *const map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        T pose[4];
        pose[0] = T(kf->pose_vector[0]);
        pose[1] = T(kf->pose_vector[1]);
        pose[2] = T(kf->pose_vector[2]);
        pose[3] = T(kf->pose_vector[3]);
        ceres::QuaternionRotatePoint(pose, map_coordinate, camera_coordinates);
        camera_coordinates[0] += (T)kf->pose_vector[4];
        camera_coordinates[1] += (T)kf->pose_vector[5];
        camera_coordinates[2] += (T)kf->pose_vector[6];
        camera_coordinates[2] = ceres::fmax(camera_coordinates[2], 1e-1);
        T inv_d = T(1) / camera_coordinates[2];
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        T disp_pred = T(kf->K(0, 0)) * kf->BASELINE * inv_d;
        T disp_meas = T(f->kpu.pt.x) - T(f->right_coordinate);      
        residuals[0] = (x - T(f->kpu.pt.x)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[1] = (y - T(f->kpu.pt.y)) / kf->POW_OCTAVE[f->kpu.octave];
        residuals[2] = (disp_pred - disp_meas) / kf->POW_OCTAVE[f->kpu.octave];
        return true;
    }

    static ceres::CostFunction *CreateFixedStereo(KeyFrame *kf, Feature *f)
    {
        return (new ceres::AutoDiffCostFunction<BundleAjustmentFixedKeyFrameStereo, 3, 3>(new BundleAjustmentFixedKeyFrameStereo(kf, f)));
    }

private:
    KeyFrame *kf;
    Feature *f;
};

class BundleAdjustment {
public:
    BundleAdjustment() {};
    void solve_ceres(Map *mapp, KeyFrame *frame);
};

#endif