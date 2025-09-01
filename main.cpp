#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "orb.hpp"

int main() {
    // Load grayscale and color images
    cv::Mat img_gs = cv::imread("1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img = cv::imread("1.jpg", cv::IMREAD_COLOR);

    if (img.empty() || img_gs.empty()) {
        std::cout << "Could not read the image." << std::endl;
        return 1;
    }

    // Detect FAST keypoints
    auto keypoints = ORB::computeFastKeypoints(img_gs, 20.0f);
    
    for(const auto& pt : keypoints) {
        std::cout << "Keypoint at (" << pt.second << ", " << pt.first << ")\n";
    }

    printf("Detected %zu keypoints\n", keypoints.size());
    // Draw detected corners
    for (const auto& pt : keypoints) {
        cv::circle(img, cv::Point(pt.second, pt.first), 3, cv::Scalar(0, 255, 0), -1);
    }

    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    // Show result
    cv::imshow("FAST Corners", img);
    cv::waitKey(0);

    return 0;
}