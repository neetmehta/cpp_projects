#pragma once
#include <vector>
#include <utility>
#include <opencv2/opencv.hpp>
#include "Keypoint.hpp"

namespace ORB
{
    extern std::vector<Keypoint> FastPixels;
    extern std::vector<Keypoint> surroundingPixels;
    float getICAngle(cv::Mat &image, ORB::Keypoint &kp, int radius, std::vector<int> umax);
    class ORBDescriptor
    {
    private:
        int patchSize = 31;
        int nkeypoints = 500;
        int nlevels = 8;
        float scaleFactor = 1.2f;
        int edgeThreshold = 31;
        int fastThreshold = 20;
        std::vector<int> umax;
        int nmsRadius = 7;
        int harrisBlockSize = 7;
        int nfeatures = 256;
        std::vector<cv::Mat> imagePyramid;

    public:
        ORBDescriptor() = default;
        ~ORBDescriptor() = default;

        ORBDescriptor(int _nkeypoints = 500, int _nlevels = 8, float _scaleFactor = 1.2, int _edgeThreshold = 31, int _fastThreshold = 20, int _patchSize = 31, int _nmsRadius = 7, int _harrisBlockSize = 7, int _nfeatures = 256);
        void getHarrisScore(const cv::Mat &image, std::vector<Keypoint> &keypoints, int ksize = 3, double k = 0.04);

        void computeFastKeypoints(cv::Mat &image, std::vector<Keypoint> &keypoints, int nlevelkeypoints, int level = 0);
        void computeBriefDescriptor(cv::Mat &image, std::vector<Keypoint> &keypoints);

        std::vector<Keypoint> computeKeypoints(cv::Mat &image);

        void buildImagePyramid(const cv::Mat &image, std::vector<cv::Mat> &imagePyramid);
        std::vector<Keypoint> nonMaxSuppression(const std::vector<Keypoint> &keypoints, int rows, int cols);

        bool fastTest(cv::Mat &img, Keypoint p);
    };

} // namespace vision
