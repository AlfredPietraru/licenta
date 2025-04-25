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


class BundleAdjustmentError
{
public:
    BundleAdjustmentError(Eigen::Matrix3d K, Eigen::Vector3d observed, double scaled_sigma, bool is_monocular) : K(K), observed(observed),
     scaled_sigma(scaled_sigma), is_monocular(is_monocular) {}

    template <typename T>
    bool operator()(const T *const pose, const T *map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        T quat[4];
        quat[0] = pose[0];
        quat[1] = pose[1];
        quat[2] = pose[2];
        quat[3] = pose[3];
        ceres::QuaternionRotatePoint(quat, map_coordinate, camera_coordinates);
        camera_coordinates[0] += pose[4];
        camera_coordinates[1] += pose[5];
        camera_coordinates[2] += pose[6];
        T inv_d = T(1) / (camera_coordinates[2] + T(1e-6));
        T x = T(K(0, 0)) * camera_coordinates[0] * inv_d + T(K(0, 2));
        T y = T(K(1, 1)) * camera_coordinates[1] * inv_d + T(K(1, 2));
        residuals[0] = (x - observed(0)) / scaled_sigma;
        residuals[1] = (y - observed(1)) / scaled_sigma;
        if (this->is_monocular)
            return true;

        T z_projected = x - T(K(0, 0)) * 0.08 * inv_d;
        residuals[2] = (z_projected - observed(2)) / scaled_sigma;
        return true;
    }

    static ceres::CostFunction *Create_Monocular(Eigen::Matrix3d K, Eigen::Vector3d observed, double scaled_sigma, bool is_monocular)
    {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentError, 2, 7, 3>(new BundleAdjustmentError(K, observed, scaled_sigma, is_monocular)));
    }

    static ceres::CostFunction *Create_Stereo(Eigen::Matrix3d K, Eigen::Vector3d observed, double scaled_sigma, bool is_monocular)
    {
        return (new ceres::AutoDiffCostFunction<BundleAdjustmentError, 3, 7, 3>(new BundleAdjustmentError(K, observed, scaled_sigma, is_monocular)));
    }

private:
    Eigen::Matrix3d K;
    Eigen::Vector3d observed;
    double scaled_sigma;
    bool is_monocular;
};


class BundleAdjustmentFixedKeyFrame {
public:
    BundleAdjustmentFixedKeyFrame(KeyFrame *kf, Eigen::Vector3d observed, double scaled_sigma, bool is_monocular) : kf(kf), observed(observed), scaled_sigma(scaled_sigma), is_monocular(is_monocular) {
        Eigen::Quaterniond quaternion = kf->Tcw.unit_quaternion();
        quat[0] = quaternion.w();
        quat[1] = quaternion.x();
        quat[2] = quaternion.y();
        quat[3] = quaternion.z();
        Eigen::Vector3d t = kf->Tcw.translation();
        translation[0] = t(0);
        translation[1] = t(1);
        translation[2] = t(2);
    }

    template <typename T>
    bool operator()(const T *map_coordinate, T *residuals) const
    {
        T camera_coordinates[3];
        ceres::QuaternionRotatePoint((T*)quat, (T*)map_coordinate, camera_coordinates);
        camera_coordinates[0] += (T)translation[4];
        camera_coordinates[1] += (T)translation[5];
        camera_coordinates[2] += (T)translation[6];
        T inv_d = T(1) / (camera_coordinates[2] + T(1e-6));
        T x = T(kf->K(0, 0)) * camera_coordinates[0] * inv_d + T(kf->K(0, 2));
        T y = T(kf->K(1, 1)) * camera_coordinates[1] * inv_d + T(kf->K(1, 2));
        residuals[0] = (x - observed(0)) / scaled_sigma;
        residuals[1] = (y - observed(1)) / scaled_sigma;
        if (this->is_monocular)
            return true;

        T z_projected = x - T(kf->K(0, 0)) * 0.08 * inv_d;
        residuals[2] = (z_projected - observed(2)) / scaled_sigma;
        return true;
        return true;
    }

    static ceres::CostFunction *Create_Monocular(KeyFrame *kf, Eigen::Vector3d observed, double scaled_sigma, bool is_monocular)
    {
        return new ceres::AutoDiffCostFunction<BundleAdjustmentFixedKeyFrame, 2, 3>(new BundleAdjustmentFixedKeyFrame(kf, observed, scaled_sigma, is_monocular));
    }

    static ceres::CostFunction *Create_Stereo(KeyFrame *kf, Eigen::Vector3d observed, double scaled_sigma, bool is_monocular)
    {
        return new ceres::AutoDiffCostFunction<BundleAdjustmentFixedKeyFrame, 3, 3>(new BundleAdjustmentFixedKeyFrame(kf, observed, scaled_sigma, is_monocular));
    }

private:
    KeyFrame *kf;
    Eigen::Vector3d observed;
    double scaled_sigma;
    bool is_monocular;
    double quat[4];
    double translation[3];
};



class BundleAdjustment {
public:
    BundleAdjustment() {};
    Sophus::SE3d solve_ceres(Map *mapp, KeyFrame *frame);
};

#endif