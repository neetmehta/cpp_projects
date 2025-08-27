#pragma once
#include <vector>
#include <pair>
#include <opencv2/opencv.hpp>

namespace vision {

std::vector<std::pair> computeFastKeypoints(cv::Mat& image, float threshold);

} // namespace vision
