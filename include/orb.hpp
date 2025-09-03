#pragma once
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include "pixel.hpp"

namespace ORB
{
    void getHarrisScore(const cv::Mat &image, std::vector<Pixel> &keypoints, int blockSize = 7, int ksize = 3, double k = 0.04);
    extern std::vector<Pixel> FastPixels;
    extern std::vector<Pixel> surroundingPixels;
    std::vector<std::pair<int, int>> computeORBKeypoints(cv::Mat &image, float fastThreshold = 20, int blockSize = 7, int edgeThreshold = 31, int patchSize = 31, int nfeatures = 500, int nlevels = 8, float scaleFactor = 1.2);
    // Multi-scale (pyramid) keypoint detection
    std::vector<std::pair<int, int>> computeORBKeypointsPyramid(cv::Mat &image,
                                                                float fastThreshold,
                                                                int blockSize,
                                                                int edgeThreshold,
                                                                int patchSize,
                                                                int nfeatures,
                                                                int nlevels,
                                                                float scaleFactor);
    std::vector<bool> nonMaxSuppression(const std::vector<Pixel> &keypoints, int radius);

    bool fastTest(cv::Mat &img, unsigned int p);

} // namespace vision
