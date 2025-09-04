#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "orb.hpp"


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
    ORB::Keypoint kp = ORB::Keypoint(994,672);
    keypoints.emplace_back(kp);
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