#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include "slam_constants.h"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;


Point2d pixel2cam(const Point2d &p, const Mat &K) {
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

int partition(vector<DMatch> &vec, int low, int high) {
    int i = (low - 1);
    for (int j = low; j <= high - 1; j++) {
        if (vec[j].distance < vec[high].distance) {
            i++;
            swap(vec[i], vec[j]);
        }
    }
    swap(vec[i + 1], vec[high]);
    return (i + 1);
}

void sort_matches_based_on_distance(vector<DMatch> &matches, int low, int high) {
    if (low < high) {
      int pi = partition(matches, low, high);
      sort_matches_based_on_distance(matches, low, pi - 1);
      sort_matches_based_on_distance(matches, pi + 1, high);
  }
}

vector<DMatch> get_feature_matches_images(Mat descriptors_1, Mat descriptors_2, std::size_t elements_selected) {
    vector<DMatch> matches;
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DESCRIPTION_MATCHER_ALGORITHM);
    cout << "pe acoloooo\n";
    Mat des1, des2;
    if (descriptors_1.type() != CV_8U) {
      descriptors_1.convertTo(des1, CV_8U);
      descriptors_2.convertTo(des2, CV_8U);
    }
    matcher->match(des1, des2, matches);
    sort_matches_based_on_distance(matches, 0, matches.size() - 1);
    vector<DMatch> good_matches(matches.begin(), matches.begin() + min(matches.size(), elements_selected));
    return good_matches;
}

std::pair<Mat, Mat> pose_estimation_2d_2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2,
                      std::vector<DMatch> matches) {
    vector<Point2f> first_image_points, second_image_points;
    for (int i=0;i< matches.size();i++)
    {
      int idx1=matches[i].trainIdx; 
      int idx2=matches[i].queryIdx;

      first_image_points.push_back(keypoints_1[idx1].pt);
      second_image_points.push_back(keypoints_2[idx2].pt);
    }   
    Mat essential_matrix;
    essential_matrix = findEssentialMat(first_image_points, second_image_points, FOCAL_LENGTH, PRINCIPAL_POINT);
    cout << "essential_matrix is " << endl << essential_matrix << endl;
    // Mat homography_matrix;
    // homography_matrix = findHomography(first_image_points, second_image_points, RANSAC, 3);
    // cout << "homography_matrix is " << endl << homography_matrix << endl;
    Mat R, t;
    recoverPose(essential_matrix, first_image_points, second_image_points, R, t, FOCAL_LENGTH, PRINCIPAL_POINT);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
    return std::pair<Mat, Mat>(R, t);
}

// using P3P method
std::pair<Mat, Mat> pose_estimation_3d_2d(std::vector<KeyPoint> keypoints_1, std::vector<KeyPoint> keypoints_2, 
    std::vector<DMatch> matches, Mat depth) {
    vector<Point3d> points_in3d;
    vector<Point2d> points_in2d;
    for (DMatch m : matches) {
      unsigned short d = depth.ptr<int>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
      if (d <= 0) continue;
      float dd = d / DEPTH_NORMALIZATION;
      Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, CAMERA_MATRIX);
      points_in3d.push_back(Point3d(p1.x * dd, p1.y * dd, dd));
      points_in2d.push_back(keypoints_2[m.trainIdx].pt);
    }
    Mat r, t;
    // pag 160
    solvePnP(points_in3d, points_in2d, CAMERA_MATRIX, Mat(), r, t, false);
    Mat R;
    Rodrigues(r, R);
    return std::pair<Mat, Mat>(R, t);

}

int main(int argc, char **argv) {
  Mat img_1 = imread("../1.png", cv::IMREAD_COLOR);
  Mat img_2 = imread("../2.png", cv::IMREAD_COLOR);
  Mat depth1 = imread("../1_depth.png", cv::IMREAD_GRAYSCALE);
  Mat depth2 = imread("../2_depth.png", cv::IMREAD_GRAYSCALE);
  assert(img_1.data != nullptr && img_2.data != nullptr);
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  Mat descriptors_1, descriptors_2;
  // Ptr<ORB> alg = ORB::create(NR_FEATURES_THRESHOLD);
  Ptr<SIFT> alg = SIFT::create(NR_FEATURES_THRESHOLD);
  alg->detect(img_1, keypoints_1);
  alg->detect(img_2, keypoints_2);
  alg->compute(img_1, keypoints_1, descriptors_1);
  alg->compute(img_2, keypoints_2, descriptors_2);
  cout << descriptors_1.type() << " " << descriptors_2.type() << "\n";
  vector<DMatch> goodMatches = get_feature_matches_images(descriptors_1, descriptors_2, NR_FEATURES_MATHCED);
  cout << goodMatches.size() << "\n";

  std::pair<Mat, Mat> movement = pose_estimation_3d_2d(keypoints_1, keypoints_2, goodMatches, depth1);
  Mat R = movement.first;
  Mat t = movement.second;
  cout << "R=" << endl << R << endl;
  cout << "t=" << endl << t << endl;
  return 0;
}