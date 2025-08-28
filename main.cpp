#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{


    // Load image
    Mat img = imread("1.jpg", IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error: Could not open image!" << endl;
        return -1;
    }

    // Create FAST detector
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(
        75,     // threshold
        true    // nonmaxSuppression
    );

    // Detect keypoints
    vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);

    // Draw keypoints
    Mat outImg;
    drawKeypoints(img, keypoints, outImg, Scalar(0, 255, 0), DrawMatchesFlags::DEFAULT);

    for(auto& kp : keypoints) {
        cout << "Keypoint at (" << kp.pt.x << ", " << kp.pt.y << ") with size " << kp.size << endl;
    }

    printf("Detected %lu keypoints\n", keypoints.size());

    cv::resize(outImg, outImg, Size(), 0.5, 0.5); // Resize for better visibility

    // Show result
    imshow("FAST Keypoints", outImg);
    waitKey(0);

    return 0;
}
