#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include "orb.hpp"

int main()
{
    // Load grayscale and color images
    cv::Mat img_gs = cv::imread("1.png", cv::IMREAD_GRAYSCALE);
    cv::Mat img_color = cv::imread("1.png", cv::IMREAD_COLOR);
    if (img_color.empty() || img_gs.empty())
    {
        std::cout << "Could not read the image." << std::endl;
        return 1;
    }
    printf("Image size: %dx%d\n", img_color.cols, img_color.rows);

    // Copy images for visualization
    cv::Mat img_custom = img_color.clone();
    cv::Mat img_builtin = img_color.clone();

    // -------------------------------
    // 1. Your ORB Keypoints
    // -------------------------------
    auto keypoints_custom = ORB::ORBDescriptor(500, 8, 1.2, 31, 20, 31).computeKeypoints(img_gs);

    for (const auto &pt : keypoints_custom)
    {
        float scale = std::pow(1.2, pt.getLevel());
        cv::circle(img_custom, cv::Point(pt.getX(), pt.getY()), 3, cv::Scalar(0, 255, 0), 1);
    }

    // -------------------------------
    // 2. OpenCV Built-in ORB Keypoints
    // -------------------------------
    std::vector<cv::KeyPoint> keypoints_builtin;
    cv::Ptr<cv::ORB> orb = cv::ORB::create(500, 1.2f, 8, 31, 0, 2, cv::ORB::HARRIS_SCORE, 31);
    orb->detect(img_gs, keypoints_builtin);

    cv::drawKeypoints(img_builtin, keypoints_builtin, img_builtin, cv::Scalar(0, 0, 255),
                      cv::DrawMatchesFlags::DEFAULT);

    // -------------------------------
    // 3. Combine images side by side
    // -------------------------------
    cv::Mat combined;
    cv::hconcat(img_custom, img_builtin, combined);
    cv::resize(combined, combined, cv::Size(), 0.5,0.5);

    // Show result
    cv::imshow("Custom ORB (left) vs OpenCV ORB (right)", combined);
    cv::waitKey(0);

    return 0;
}
