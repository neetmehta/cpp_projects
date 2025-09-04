#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "orb.hpp"

std::vector<int> getBounds(int patchSize)
{
     int halfPatchSize = patchSize / 2;
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Make sure we are symmetric
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }
    return umax;
}

static float IC_Angle(const cv::Mat &image, const int half_k, cv::Point2f pt,
                      const std::vector<int> &u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar *center = &image.at<uchar>(cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -half_k; u <= half_k; ++u)
    {
        m_10 += u * center[u];
        // printf("[CV] %d\n",center[u]);
    }
    printf("\n[CV] m01=%d and m10=%d\n", m_01, m_10);
    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= half_k; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v * step], val_minus = center[u - v * step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }
    
    return cv::fastAtan2((float)m_01, (float)m_10);
}

float getICangle(cv::Mat &image, ORB::Keypoint &kp, int radius, std::vector<int> umax)
{
    int m_01 = 0, m_10 = 0;
    int x0 = kp.getX();
    int y0 = kp.getY();

    for (int u = -radius; u <= radius; ++u)
    {
        m_10 += image.at<uchar>(y0, x0 + u) * u;
        // printf("[MY] %d\n", image.at<uchar>(y0, x0 + u));
    }
    

    for (int v = 1; v <= radius; v++)
    {
        int d = umax[v];
        for (int u = -d; u <= d; u++)
        {
            m_10 += image.at<uchar>(y0+v, x0+u)*u;
            m_10 += image.at<uchar>(y0-v, x0+u)*u;

            m_01 += image.at<uchar>(y0+v, x0+u)*v;
            m_01 += image.at<uchar>(y0-v, x0+u)*(-v);
        }
    }
    
    return cv::fastAtan2((float)m_01, (float)m_10);
}

int main()
{
    // Load grayscale and color images
    cv::Mat img_gs = cv::imread("1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img = cv::imread("1.jpg", cv::IMREAD_COLOR);
    printf("Image size: %dx%d\n", img.cols, img.rows);
    if (img.empty() || img_gs.empty())
    {
        std::cout << "Could not read the image." << std::endl;
        return 1;
    }

    // Detect FAST keypoints
    // auto keypoints = ORB::computeKeypoints(img_gs, 20.0f, 7, 31, 31, 500, 8, 1.2);
    std::vector<ORB::Keypoint> keypoints;
    ORB::Keypoint kp = ORB::Keypoint(994, 672);
    keypoints.emplace_back(kp);
    cv::Point2f pt = cv::Point2f(994, 672);
    int patchSize = 31;
    int halfPatchSize = patchSize/2;
    std::vector<int> umax = getBounds(patchSize);
    float cvAngle = IC_Angle(img_gs, halfPatchSize, pt, umax);
    float myAngle = getICangle(img_gs, kp, halfPatchSize, umax);

    printf("cvAngle is: %f and myAngle is: %f", cvAngle, myAngle);
    // Draw detected corners
    for (const auto &pt : keypoints)
    {
        printf("Keypoint at (%d, %d) with harris score %.7f\n", pt.getX(), pt.getY(), pt.getScore());
        cv::circle(img, cv::Point(pt.getX() * std::pow(1.2, pt.getLevel()), pt.getY() * std::pow(1.2, pt.getLevel())), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    // Show result
    cv::imshow("FAST Corners", img);
    cv::waitKey(0);

    return 0;
}