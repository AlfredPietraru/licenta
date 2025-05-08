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

#include "../include/MapDrawer.h"
#include "../include/MapPoint.h"
#include "../include/KeyFrame.h"
#include <pangolin/pangolin.h>

MapDrawer::MapDrawer(Map *pMap) : mapp(pMap)
{
    mKeyFrameSize = 0.05;
    mKeyFrameLineWidth = 3;
    mGraphLineWidth = 0.9;
    mPointSize = 4;
    mCameraSize = 0.08;
    mCameraLineWidth = 3;

    float mViewpointX =  0.0;
    float mViewpointY = -0.2;
    float mViewpointZ = -0.6;
    float mViewpointF =  800;

    pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);
    glEnable(GL_DEPTH_TEST);
    // 3D Mouse handler requires depth testing to be enabled

    // Issue specific OpenGl we might need
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    this->s_cam = pangolin::OpenGlRenderState(
        pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
        pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));

    // Add named OpenGL viewport to window and provide 3D Handler
    this->d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
                                .SetHandler(new pangolin::Handler3D(s_cam));
    Twc.SetIdentity();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    s_cam.Follow(Twc);
    d_cam.Activate(s_cam);
    translation_pose = Sophus::SE3d(Eigen::Quaterniond(-0.3266, 0.6583, 0.6112, -0.2938), Eigen::Vector3d(1.3434, 0.6271, 1.6606));
}

void MapDrawer::run(KeyFrame *kf, cv::Mat frame, Sophus::SE3d grountruth_pose)
{
    int wait_time = kf->current_idx < 180 ? 30 : 30;
    kf->debug_keyframe(frame, wait_time);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    Eigen::Matrix3d Rwc = kf->Tcw.inverse().rotationMatrix();
    Eigen::Vector3d twc = kf->Tcw.inverse().translation();
    this->GetCurrentOpenGLCameraMatrix(Rwc, twc);
    s_cam.Follow(Twc);
    d_cam.Activate(s_cam);
    if (!kf->isKeyFrame) {
        current_pose.push_back(kf->camera_center_world);
    }

    Sophus::SE3d relative_pose = translation_pose.inverse() * grountruth_pose;
    Eigen::Vector3d center_groundtruth = -relative_pose.rotationMatrix().transpose() * relative_pose.translation();
    center_groundtruth(2) = -center_groundtruth(2);
    this->current_groundtruth.push_back(center_groundtruth);
    this->DrawKeyFrames();
    // this->DrawRegularFrames();
    this->DrawMapPoints();
    this->DrawKeyFramesConnections();
    pangolin::FinishFrame();
}



void MapDrawer::DrawKeyFramesConnections() {
    for (int i = 0; i < (int)mapp->keyframes.size(); i++) {
        KeyFrame *kfi = mapp->keyframes[i];
        KeyFrame *best_kf = mapp->spanning_tree[kfi];
        if (best_kf == nullptr) continue;
        Eigen::Vector3d pi = kfi->camera_center_world;
        Eigen::Vector3d pj = best_kf->camera_center_world;
        glColor3f(0.0f, 0.0f, 0.0f);
        glLineWidth(mKeyFrameLineWidth);
        glBegin(GL_LINES);
        glVertex3f(pi.x(), pi.y(), pi.z());
        glVertex3f(pj.x(), pj.y(), pj.z());
        glEnd();
    }
}

void MapDrawer::draw_frame_pose(Eigen::Vector3d p, double red, double green, double blue) {
    const float &w = mKeyFrameSize;
    const float h = w * 0.75;
    const float z = w * 0.6;
    glColor3f(red, green, blue);
    glLineWidth(mKeyFrameLineWidth);
    glBegin(GL_LINES);
    glVertex3f(p.x(), p.y(), p.z());
    glVertex3f(p.x() + w, p.y() + h, p.z() + z);
    glVertex3f(p.x(), p.y(), p.z());
    glVertex3f(p.x() + w, p.y() - h, p.z() + z);
    glVertex3f(p.x(), p.y(), p.z());
    glVertex3f(p.x() - w, p.y() - h, p.z() + z);
    glVertex3f(p.x(), p.y(), p.z());
    glVertex3f(p.x() - w, p.y() + h, p.z() + z);
    glVertex3f(p.x() + w, p.y() + h, p.z() + z);
    glVertex3f(p.x() + w, p.y() - h, p.z() + z);
    glVertex3f(p.x() - w, p.y() + h, p.z() + z);
    glVertex3f(p.x() - w, p.y() - h, p.z() + z);
    glVertex3f(p.x() - w, p.y() + h, p.z() + z);
    glVertex3f(p.x() + w, p.y() + h, p.z() + z);
    glVertex3f(p.x() - w, p.y() - h, p.z() + z);
    glVertex3f(p.x() + w, p.y() - h, p.z() + z);
    glEnd();
}

void MapDrawer::DrawKeyFrames()
{
    for (KeyFrame *kf : mapp->keyframes)
        draw_frame_pose(kf->camera_center_world, 0.0f, 1.0f, 0.0f);
}

void MapDrawer::DrawRegularFrames() {
    for (Eigen::Vector3d p : this->current_pose) {
        draw_frame_pose(p, 0.0f, 0.3f, 1.0f);
    }
    for (Eigen::Vector3d p : this->current_groundtruth) {
        draw_frame_pose(p, 0.0f, 0.3f, 0.5f);
    }
}


void MapDrawer::GetCurrentOpenGLCameraMatrix(Eigen::Matrix3d Rwc, Eigen::Vector3d twc)
{
    Twc.m[0]  = Rwc(0, 0);
    Twc.m[1]  = Rwc(1, 0);
    Twc.m[2]  = Rwc(2, 0);
    Twc.m[3]  = 0.0;
    Twc.m[4]  = Rwc(0, 1);
    Twc.m[5]  = Rwc(1, 1);
    Twc.m[6]  = Rwc(2, 1);
    Twc.m[7]  = 0.0;
    Twc.m[8]  = Rwc(0, 2);
    Twc.m[9]  = Rwc(1, 2);
    Twc.m[10] = Rwc(2, 2);
    Twc.m[11] = 0.0;
    Twc.m[12] = twc(0);
    Twc.m[13] = twc(1);
    Twc.m[14] = twc(2);
    Twc.m[15] = 1.0;
}

void MapDrawer::DrawMapPoints()
{
    const std::unordered_set<MapPoint *> &all_map_points = mapp->get_all_map_points();
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (MapPoint *mp : all_map_points)
    {  
        glVertex3f(mp->wcoord_3d(0), mp->wcoord_3d(1), mp->wcoord_3d(2));
    }
    glEnd();
}