#include "../include/Common.h"


double Common::get_rgbd_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* feature, double chi2) {
    double residuals[3];
    Eigen::Vector4d camera_coordinates = kf->mat_camera_world * mp->wcoord;
    if (camera_coordinates[2] <= 1e-1) return 100000;
    double inv_d = 1 / camera_coordinates[2];
    double x = kf->K(0, 0) * camera_coordinates[0] * inv_d + kf->K(0, 2);
    double y = kf->K(1, 1) * camera_coordinates[1] * inv_d + kf->K(1, 2);
    cv::KeyPoint kpu = feature->kpu;
    residuals[0] = (x - kpu.pt.x) * kf->INVERSE_POW_OCTAVE[kpu.octave];
    residuals[1] = (y - kpu.pt.y) * kf->INVERSE_POW_OCTAVE[kpu.octave]; 

    double disp_pred = kf->K(0, 0) * 0.08 * inv_d;
    double disp_meas = kpu.pt.x - feature->right_coordinate;      
    residuals[2] = (disp_pred - disp_meas) * kf->INVERSE_POW_OCTAVE[kpu.octave];
    double a = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1] + residuals[2] * residuals[2]);
    if (a <= sqrt(chi2)) return pow(a, 2) / 2;
    return sqrt(chi2) * a - chi2 / 2;
}

double Common::get_monocular_reprojection_error(KeyFrame *kf, MapPoint *mp, Feature* feature, double chi2) {
    double residuals[2];
    Eigen::Vector4d camera_coordinates = kf->mat_camera_world * mp->wcoord;
    if (camera_coordinates[2] <= 1e-1) return 100000;
    double inv_d = 1 / camera_coordinates[2];
    double x = kf->K(0, 0) * camera_coordinates[0] * inv_d + kf->K(0, 2);
    double y = kf->K(1, 1) * camera_coordinates[1] * inv_d + kf->K(1, 2);
    cv::KeyPoint kpu = feature->kpu;
    residuals[0] = (x - kpu.pt.x) * kf->INVERSE_POW_OCTAVE[kpu.octave];
    residuals[1] = (y - kpu.pt.y) * kf->INVERSE_POW_OCTAVE[kpu.octave];
    double a = sqrt(residuals[0] * residuals[0] + residuals[1] * residuals[1]);
    if (a <= sqrt(chi2)) return pow(a, 2) / 2;
    return sqrt(chi2) * a - chi2 / 2;
}