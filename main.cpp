#include <iostream>
#include <ceres/ceres.h>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "slam_constants.h"
#include "rotation.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace ceres;
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;

bool isRotationMatrix(const cv::Mat &R)
{
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
    return cv::norm(I, shouldBeIdentity) < 1e-6;
}

cv::Vec3d rotationMatrixToEulerAngles(const cv::Mat &R)
{

     assert(isRotationMatrix(R));
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) +
                         R.at<double>(1, 0) * R.at<double>(1, 0));

    bool singular = sy < 1e-6; // If sy is close to zero, the matrix is singular

    double x, y, z;
    if (!singular)
    {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    }
    else
    {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}


Point2d pixel2cam(const Point2d &p, const Mat &K)
{
  return Point2d(
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1));
}

int partition(vector<DMatch> &vec, int low, int high)
{
  int i = (low - 1);
  for (int j = low; j <= high - 1; j++)
  {
    if (vec[j].distance < vec[high].distance)
    {
      i++;
      swap(vec[i], vec[j]);
    }
  }
  swap(vec[i + 1], vec[high]);
  return (i + 1);
}

void sort_matches_based_on_distance(vector<DMatch> &matches, int low, int high)
{
  if (low < high)
  {
    int pi = partition(matches, low, high);
    sort_matches_based_on_distance(matches, low, pi - 1);
    sort_matches_based_on_distance(matches, pi + 1, high);
  }
}

vector<DMatch> get_feature_matches_images(Mat descriptors_1, Mat descriptors_2, std::size_t elements_selected)
{
  vector<DMatch> matches;
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DESCRIPTION_MATCHER_ALGORITHM);
  matcher->match(descriptors_1, descriptors_2, matches);
  sort_matches_based_on_distance(matches, 0, matches.size() - 1);
  vector<DMatch> good_matches(matches.begin(), matches.begin() + min(matches.size(), elements_selected));
  return good_matches;
}

std::pair<Mat, Mat> pose_estimation_2d_2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2,
                                          std::vector<DMatch> matches)
{
  vector<Point2f> first_image_points, second_image_points;
  for (int i = 0; i < matches.size(); i++)
  {
    int idx1 = matches[i].queryIdx;
    int idx2 = matches[i].trainIdx;
    first_image_points.push_back(keypoints_1[idx1].pt);
    second_image_points.push_back(keypoints_2[idx2].pt);
  }
  Mat essential_matrix;
  essential_matrix = findEssentialMat(first_image_points, second_image_points, FOCAL_LENGTH, PRINCIPAL_POINT);
  cout << "essential_matrix is " << endl
       << essential_matrix << endl;
  // Mat homography_matrix;
  // homography_matrix = findHomography(first_image_points, second_image_points, RANSAC, 3);
  // cout << "homography_matrix is " << endl << homography_matrix << endl;
  Mat R, t;
  recoverPose(essential_matrix, first_image_points, second_image_points, R, t, FOCAL_LENGTH, PRINCIPAL_POINT);
  cout << "R is " << endl
       << R << endl;
  cout << "t is " << endl
       << t << endl;
  return std::pair<Mat, Mat>(R, t);
}

// using P3P method
std::pair<Mat, Mat> pose_estimation_3d_2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2,
                                          std::vector<DMatch> matches, Mat depth)
{
  vector<Point3d> points_in3d;
  vector<Point2d> points_in2d;
  cout << matches.size() << "\n";
  for (DMatch m : matches)
  {
    unsigned short d = depth.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d <= 0)
      continue;
    double dd = d / DEPTH_NORMALIZATION;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, CAMERA_MATRIX);
    points_in3d.push_back(Point3d(p1.x * dd, p1.y * dd, dd));
    points_in2d.push_back(keypoints_2[m.trainIdx].pt);
  }
  cout << points_in2d.size() << " " << points_in3d.size() << "\n";
  Mat r, t;
  // pag 160
  solvePnP(points_in3d, points_in2d, CAMERA_MATRIX, Mat(), r, t, false);
  Mat R;
  Rodrigues(r, R);
  return std::pair<Mat, Mat>(R, t);
}

struct SnavleyReprojectionError
{
  SnavleyReprojectionError(double observed_x, double observed_y) : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T *const camera, const T *const point, T *residuals) const
  {
    T p[3];
    //   p = R * (point)
    ceres::AngleAxisRotatePoint(camera, point, p);
    // applied transtlation p = R * (point) + t;
    p[0] += camera[3];
    p[1] += camera[4];
    p[2] += camera[5];
    // normalization
    T xp = p[0] / p[2];
    T yp = p[1] / p[2];

    const T &l1 = camera[7];
    const T &l2 = camera[8];
    T r2 = xp * xp + yp * yp;
    T distortion = 1.0 + r2 * (l1 + l2 * r2);
    // Compute final projected point position.
    const T &focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;
    residuals[0] = predicted_x - observed_x;
    residuals[1] = predicted_y - observed_y;
    return true;
  }

  static CostFunction *Create(double observed_x, double observed_y)
  {
    return new AutoDiffCostFunction<SnavleyReprojectionError, 2, 9, 3>(new SnavleyReprojectionError(observed_x, observed_y));
  }

private:
  double observed_x;
  double observed_y;
};

int main()
{
  Mat img_1 = imread("../1.png", cv::IMREAD_COLOR);
  Mat img_2 = imread("../2.png", cv::IMREAD_COLOR);
  Mat depth1 = imread("../1_depth.png", cv::IMREAD_GRAYSCALE);
  Mat depth2 = imread("../2_depth.png", cv::IMREAD_GRAYSCALE);
  assert(img_1.data != nullptr && img_2.data != nullptr);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  Ptr<ORB> alg = ORB::create(NR_FEATURES_THRESHOLD, 1.2F, 8, 25, 0, 2, cv::ORB::HARRIS_SCORE, 5, 20);
  alg->detect(img_1, keypoints_1);
  alg->detect(img_2, keypoints_2);
  alg->compute(img_1, keypoints_1, descriptors_1);
  alg->compute(img_2, keypoints_2, descriptors_2);
  cout << descriptors_1.size() << " " << descriptors_2.size() << "\n";
  vector<DMatch> goodMatches = get_feature_matches_images(descriptors_1, descriptors_2, NR_FEATURES_MATHCED);
  cout << goodMatches.size() << " matches \n";
  vector<Point2d> points_in2d;
  vector<Point3d> points_in3d;
  cout << goodMatches.size() << "\n";
  for (DMatch m : goodMatches)
  {
    unsigned short d = depth1.ptr<int>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
    if (d <= 0)
      continue;
    double dd = d / DEPTH_NORMALIZATION;
    Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, CAMERA_MATRIX);
    points_in3d.push_back(Point3d(p1.x * dd, p1.y * dd, dd));
    points_in2d.push_back(keypoints_2[m.trainIdx].pt);
  }
  
  cout << points_in2d.size() << " " << points_in3d.size() << "\n";
  Mat r, t, R;
  // pag 160
  solvePnP(points_in3d, points_in2d, CAMERA_MATRIX, Mat(), r, t, false);
  cv::Rodrigues(r, R);
  cv::Vec3d rvec = rotationMatrixToEulerAngles(R);
  cout << rvec << "\n";
  cout << t << "\n";
  double camera[9] = {rvec[0], rvec[1], rvec[2],
         t.at<double>(0, 0), t.at<double>(1, 0), t.at<double>(2, 0), 520.9, 0, 0};
  for (int i = 0; i < 9; i++) {
    cout << camera[i] << " ";
  }
  cout << "\n";
  Problem problem;
  vector<vector<double>> converted;
  for (int i = 0; i < points_in3d.size(); i++) {
    converted.push_back({points_in3d[i].x, points_in3d[i].y, points_in3d[i].z});
    CostFunction *cost_function = SnavleyReprojectionError::Create(points_in2d[i].x, points_in2d[i].y);
    // ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);
    problem.AddResidualBlock(cost_function, nullptr, camera, converted.back().data());
  }
  
  Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n\n\n";
  cout << "ajunge aici\n";
  for (int i = 0; i < 9; i++) {
    cout << camera[i] << " ";
  }
  cout << "\n";
   return 0;
}