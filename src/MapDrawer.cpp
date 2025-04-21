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

MapDrawer::MapDrawer(Map *pMap, Eigen::Matrix4d T) : mapp(pMap)
{

    mKeyFrameSize = 0.05;
    mKeyFrameLineWidth = 1;
    mGraphLineWidth = 0.9;
    mPointSize = 2;
    mCameraSize = 0.08;
    mCameraLineWidth = 3;

    float mViewpointX = 0;
    float mViewpointY = -0.7;
    float mViewpointZ = -1.8;
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

    cv::Mat mat;
    cv::eigen2cv(T, mat);
    Twc.SetIdentity();
}

void MapDrawer::run(cv::Mat T, std::unordered_set<MapPoint *> &local_map)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    T.at<float>(0, 0) += 1; // DE STERS
    this->GetCurrentOpenGLCameraMatrix(Twc);
    s_cam.Follow(Twc);
    d_cam.Activate(s_cam);
    glClearColor(1.0f,1.0f,1.0f,1.0f);
    this->DrawCurrentCamera(Twc);
    this->DrawKeyFrames(true, true);
    this->DrawMapPoints(local_map);
    pangolin::FinishFrame();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    if (bDrawKF)
    {
        for (size_t i = 0; i < mapp->keyframes.size(); i++)
        {
            KeyFrame *kf = mapp->keyframes[i];
            cv::Mat mat;
            cv::eigen2cv(kf->Tcw.matrix(), mat);

            glPushMatrix();

            glMultMatrixf(mat.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f, 0.0f, 1.0f);
            glBegin(GL_LINES);
            glVertex3f(0, 0, 0);
            glVertex3f(w, h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, -h, z);
            glVertex3f(0, 0, 0);
            glVertex3f(-w, h, z);

            glVertex3f(w, h, z);
            glVertex3f(w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(-w, -h, z);

            glVertex3f(-w, h, z);
            glVertex3f(w, h, z);

            glVertex3f(-w, -h, z);
            glVertex3f(w, -h, z);
            glEnd();

            glPopMatrix();
        }
    }
    if (bDrawGraph) {
        std::cout << "O SA DESENEZE SI GRAPHUL LA UN MOMENT DAT\n";
    }
}


void MapDrawer::DrawConsecutiveFrames() {
    const float &w = mKeyFrameSize;
    const float h = w * 0.75;
    const float z = w * 0.6;
    if (matrix_poses.size() == 0) return;
    for (int i = 0; i < (int)matrix_poses.size(); i++) {
        cv::Mat mat = matrix_poses[i];
        glPushMatrix();
        glMultMatrixf(mat.ptr<GLfloat>(0));
        glLineWidth(mKeyFrameLineWidth);
        glColor3f(0.0f, 1.0f, 0.0f);
        glBegin(GL_LINES);
        glVertex3f(0, 0, 0);
        glVertex3f(w, h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, -h, z);
        glVertex3f(0, 0, 0);
        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);
        glVertex3f(w, -h, z);
        glVertex3f(-w, h, z);
        glVertex3f(-w, -h, z);
        glVertex3f(-w, h, z);
        glVertex3f(w, h, z);
        glVertex3f(-w, -h, z);
        glVertex3f(w, -h, z);
        glEnd();
        glPopMatrix();
    }
}


void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w * 0.75;
    const float z = w * 0.6;

    glPushMatrix();

#ifdef HAVE_GLES
    glMultMatrixf(Twc.m);
#else
    glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f, 1.0f, 0.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    mCameraPose = Tcw.clone();
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if (!mCameraPose.empty())
    {
        cv::Mat Rwc(3, 3, CV_32F);
        cv::Mat twc(3, 1, CV_32F);
        {
            Rwc = mCameraPose.rowRange(0, 3).colRange(0, 3).t();
            twc = -Rwc * mCameraPose.rowRange(0, 3).col(3);
        }

        M.m[0] = Rwc.at<float>(0, 0);
        M.m[1] = Rwc.at<float>(1, 0);
        M.m[2] = Rwc.at<float>(2, 0);
        M.m[3] = 0.0;

        M.m[4] = Rwc.at<float>(0, 1);
        M.m[5] = Rwc.at<float>(1, 1);
        M.m[6] = Rwc.at<float>(2, 1);
        M.m[7] = 0.0;

        M.m[8] = Rwc.at<float>(0, 2);
        M.m[9] = Rwc.at<float>(1, 2);
        M.m[10] = Rwc.at<float>(2, 2);
        M.m[11] = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15] = 1.0;
    }
    else
        M.SetIdentity();
}


void MapDrawer::DrawMapPoints(std::unordered_set<MapPoint *> &local_map)
{
    const std::unordered_set<MapPoint *> &all_map_points = mapp->get_all_map_points();
    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0, 0.0, 0.0);

    for (auto it = all_map_points.begin(); it != all_map_points.end(); it++)
    {
        MapPoint *mp = *it;
        if (local_map.find(mp) != local_map.end())
            continue;
        glVertex3f(mp->wcoord_3d(0), mp->wcoord_3d(1), mp->wcoord_3d(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0, 0.0, 0.0);

    for (auto it = local_map.begin(); it != local_map.end(); it++)
    {
        MapPoint *mp = *it;
        glVertex3f(mp->wcoord_3d(0), mp->wcoord_3d(1), mp->wcoord_3d(2));
    }

    glEnd();
}