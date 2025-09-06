#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "orb.hpp"


int main()
{
    // Load grayscale and color images
    cv::Mat img_gs = cv::imread("1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img = cv::imread("1.png", cv::IMREAD_COLOR);
    printf("Image size: %dx%d\n", img.cols, img.rows);
    if (img.empty() || img_gs.empty())
    {
        std::cout << "Could not read the image." << std::endl;
        return 1;
    }

    // Detect FAST keypoints
    auto keypoints = ORB::ORBDescriptor(500, 8, 1.2, 31, 20, 31).computeKeypoints(img_gs);

    // Draw detected corners
    for (const auto &pt : keypoints)
    {
        printf("Keypoint at (%d, %d) with harris score %.7f\n", pt.getX(), pt.getY(), pt.getScore());
        float scale = std::pow(1.2, pt.getLevel());
        cv::circle(img, cv::Point(pt.getX(), pt.getY()), 15 * scale, cv::Scalar(0, 255, 0), 1);
    }

    int scale = 1;
    cv::resize(img, img, cv::Size(), scale, scale);
    // Show result
    cv::imshow("FAST Corners", img);
    cv::waitKey(0);

    return 0;
}