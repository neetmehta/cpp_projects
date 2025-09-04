#pragma once
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include "pixel.hpp"

namespace ORB
{
    void getHarrisScore(const cv::Mat &image, std::vector<Keypoint> &keypoints, int blockSize = 7, int ksize = 3, double k = 0.04);
    void getICAngle(const cv::Mat &image, std::vector<Keypoint> &keypoints, std::vector<int> umax, int patchSize = 31);
    extern std::vector<Keypoint> FastPixels;
    extern std::vector<Keypoint> surroundingPixels;
    void computeFastKeypoints(cv::Mat &image, std::vector<Keypoint> &keypoints, float fastThreshold = 20, int HarrisblockSize = 7, int nfeatures = 500);

    std::vector<std::pair<int, int>> computeKeypoints(cv::Mat &image, float fastThreshold = 20, int blockSize = 7, int edgeThreshold = 31, int patchSize = 31, int nfeatures = 500, int nlevels = 8, float scaleFactor = 1.2);

    void buildImagePyramid(const cv::Mat &image, std::vector<cv::Mat> &imagePyramid, int nlevels = 8, float scaleFactor = 1.2);
    std::vector<bool> nonMaxSuppression(const std::vector<Keypoint> &keypoints, int radius);

    bool fastTest(cv::Mat &img, unsigned int p);

} // namespace vision
