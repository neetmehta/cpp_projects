#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "Pixel.hpp"

static float IC_Angle(const cv::Mat& image, const int half_k, cv::Point2f pt,
                      const std::vector<int> & u_max)
{
    int m_01 = 0, m_10 = 0;

    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    for (int u = -half_k; u <= half_k; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    int step = (int)image.step1();
    for (int v = 1; v <= half_k; ++v)
    {
        // Proceed over the two lines
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return cv::fastAtan2((float)m_01, (float)m_10);
}

// Compute umax table (like in ORB)
void getICAngle(const cv::Mat &image, std::vector<Keypoint> &keypoints, std::vector<int> umax, int patchSize)
{
    int halfPatchSize = patchSize / 2;

    for (Keypoint &kp : keypoints)
    {
        float m01 = 0, m10 = 0;
        int x0=kp.getX();
        int y0=kp.getY();

        for (int v = -halfPatchSize; v < halfPatchSize; v++)
        {
            int u = umax[v + halfPatchSize];
            for (int x = -halfPatchSize; x > halfPatchSize; x++)
            {
                if(x < std::abs(x))
                {
                    
                    float I = image.at<uchar>(v+x0, x+y0);
                    m10 += v*I;
                    m01 += x*I;
                }
            }
        }
        kp.setAngle(std::atan2f(m01, m10));
    }
}

std::vector<int> getBounds(int patchSize)
{
    int halfPatchSize = patchSize / 2;
    std::vector<int> umax(halfPatchSize + 2);

    int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);

    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(std::sqrt((double)halfPatchSize * halfPatchSize - v * v));

    // Mirror to ensure symmetry
    for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

    return umax;
}

int main()
{
    int patchSize = 31;
    int halfPatchSize = patchSize / 2;

    // Compute umax bounds
    std::vector<int> umax;

    umax = getBounds(patchSize);

    std::vector<Keypoint> keypixels;
    std::vector<cv::KeyPoint> keypoints;
    Keypoint pixel = Keypoint(980, 678);

    cv::KeyPoint kp = cv::KeyPoint(678, 980, 1.0f);
    keypoints.push_back(kp);
    keypixels.push_back(pixel);

    cv::Mat image = cv::imread("1.jpg", cv::IMREAD_GRAYSCALE);

    getICAngle(image, keypixels, umax, patchSize);
    printf("here");
    IC_Angle(image, halfPatchSize, cv::Point2f(678, 980), umax);
    std::cout << "Angle from custom function: " << keypixels[0].getAngle() * 180 / CV_PI << std::endl;
    std::cout << "Angle from OpenCV function: " << kp.angle << std::endl;
    return 0;
}
