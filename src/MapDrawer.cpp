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

    mKeyFrameSize = 0.1;
    mKeyFrameLineWidth = 1;
    mGraphLineWidth = 0.9;
    mPointSize = 3;
    mCameraSize = 0.08;
    mCameraLineWidth = 3;

    float mViewpointX = 0;
    float mViewpointY = -0.7;
    float mViewpointZ = -1.0;
    float mViewpointF = 500;

    pangolin::CreateWindowAndBind("ORB-SLAM2: Map Viewer", 1024, 768);
    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

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
}

void MapDrawer::run(KeyFrame *kf, bool is_keyframe)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Eigen::Matrix3d Rwc = kf->Tcw.inverse().rotationMatrix();
    Eigen::Vector3d twc = kf->Tcw.inverse().translation();
    this->GetCurrentOpenGLCameraMatrix(Rwc, twc);
    s_cam.Follow(Twc);
    d_cam.Activate(s_cam);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    this->DrawKeyFrames();
    if (!is_keyframe) {
        FrameSimulation frameSimulation(kf->camera_center_world);
        regular_frames.push_back(frameSimulation);
    }
    // this->DrawRegularFrames();
    this->DrawMapPoints();
    pangolin::FinishFrame();
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
        draw_frame_pose(kf->camera_center_world, 1.0f, 1.0f, 1.0f);
}


void MapDrawer::DrawRegularFrames() {
    for (FrameSimulation frameSimulation : regular_frames) {
        this->draw_frame_pose(frameSimulation.camera_center, 0.0f, 0.0f, 1.0f);
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

    // glPointSize(mPointSize);
    // glBegin(GL_POINTS);
    // glColor3f(1.0, 0.0, 1.0);

    // for (MapPoint *mp : mapp->local_map)
    // {
    //     glVertex3f(mp->wcoord_3d(0), mp->wcoord_3d(1), mp->wcoord_3d(2));
    // }

    glEnd();
}