#pragma once
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include "pixel.hpp"

namespace vision
{

    extern std::vector<Pixel> FastPixels;
    extern std::vector<Pixel> surroundingPixels;
    std::vector<std::pair<int, int>> computeFastKeypoints(cv::Mat &image, float threshold);
    std::vector<bool> nonMaxSuppression(const std::vector<Pixel> &keypoints, int radius);

    bool fastTest(cv::Mat &img, unsigned int p);

} // namespace vision
