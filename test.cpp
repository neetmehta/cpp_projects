#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

int main(int argc, char** argv)
{

    // Load grayscale image
    cv::Mat img = cv::imread("1.jpg", cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // Create ORB detector
    int nfeatures     = 500;     // number of keypoints to retain
    float scaleFactor = 1.2f;    // pyramid decimation ratio
    int nlevels       = 8;       // number of pyramid levels
    int edgeThreshold = 31;      // size of border where features are not detected
    int firstLevel    = 0;       // level of pyramid to put source image
    int WTA_K         = 2;       // number of points that produce each element of the descriptor
    int scoreType     = cv::ORB::HARRIS_SCORE; // ORB::HARRIS_SCORE or ORB::FAST_SCORE
    int patchSize     = 31;      // patch size used by orientation
    int fastThreshold = 20;      // FAST threshold

    cv::Ptr<cv::ORB> orb = cv::ORB::create(
        nfeatures, scaleFactor, nlevels, edgeThreshold,
        firstLevel, WTA_K, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);

    // Detect keypoints and compute descriptors
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    std::cout << "Detected " << keypoints.size() << " keypoints." << std::endl;

    // Draw keypoints
    cv::Mat img_keypoints;
    cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar(0, 255, 0));

    cv::resize(img_keypoints, img_keypoints, cv::Size(), 0.5, 0.5);

    // Show result
    cv::imshow("ORB Keypoints", img_keypoints);
    cv::waitKey(0);

    return 0;
}
