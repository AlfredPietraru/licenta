/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>

#include<mutex>


class MapDrawer
{
public:
    MapDrawer(Map* pMap);
    Map* mapp;
    std::unordered_set<MapPoint *> all_map_points;
    
    void run(KeyFrame *kf, cv::Mat frame);
    void DrawMapPoints(bool isKeyframe);
    void DrawKeyFrames();
    void draw_frame_pose(Eigen::Vector3d p, double red, double green, double blue);
    void DrawKeyFramesConnections();
    void GetCurrentOpenGLCameraMatrix(Eigen::Matrix3d Rwc, Eigen::Vector3d twc);

private:

    int add = 0;
    Eigen::Vector3d current_pose;
    std::vector<Eigen::Vector3d> current_groundtruth;
    Sophus::SE3d translation_pose;
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
    pangolin::OpenGlMatrix Twc;
    pangolin::View d_cam;
    pangolin::OpenGlRenderState s_cam;
};
#endif // MAPDRAWER_H
