#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp> // For cv::KeyPoint
#include <opencv2/core/utility.hpp> // For cv::AutoBuffer
#include <opencv2/imgproc.hpp> // For cv::cornerHarris
#include "Pixel.hpp"
// ...existing code...


static void HarrisResponses(const cv::Mat& img, std::vector<cv::KeyPoint>& pts, int blockSize, float harris_k)
{
    CV_Assert( img.type() == CV_8UC1 && blockSize*blockSize <= 2048 );

    size_t ptidx, ptsize = pts.size();

    const uchar* ptr00 = img.ptr<uchar>();
    int step = (int)(img.step/img.elemSize1());
    int r = blockSize/2;

    float scale = (1 << 2) * blockSize * 255.0f;
    scale = 1.0f / scale;
    float scale_sq_sq = scale * scale * scale * scale;

    cv::AutoBuffer<int> ofsbuf(blockSize*blockSize);
    int* ofs = ofsbuf;
    for( int i = 0; i < blockSize; i++ )
        for( int j = 0; j < blockSize; j++ ){
            ofs[i*blockSize + j] = (int)(i*step + j);
        }

    for( ptidx = 0; ptidx < ptsize; ptidx++ )
    {
        int x0 = cvRound(pts[ptidx].pt.x - r);
        int y0 = cvRound(pts[ptidx].pt.y - r);

        const uchar* ptr0 = ptr00 + y0*step + x0;
        int a = 0, b = 0, c = 0;

        for( int k = 0; k < blockSize*blockSize; k++ )
        {
            const uchar* ptr = ptr0 + ofs[k];
            int Ix = (ptr[1] - ptr[-1])*2 + (ptr[-step+1] - ptr[-step-1]) + (ptr[step+1] - ptr[step-1]);
            int Iy = (ptr[step] - ptr[-step])*2 + (ptr[step-1] - ptr[-step-1]) + (ptr[step+1] - ptr[-step+1]);
            a += Ix*Ix;
            b += Iy*Iy;
            c += Ix*Iy;

            // printf("Debug cv: a=%d, b=%d, c=%d at x=%d, y=%d, Ix=%d, Iy=%d\n", a, b, c, x0 + (ofs[k] % step), y0 + (ofs[k] / step), Ix, Iy);
        }
        // printf("Debug: a=%d, b=%d, c=%d at (%f, %f)\n", a, b, c, pts[ptidx].pt.x, pts[ptidx].pt.y);
        std::cout << "Keypoint at (" << pts[ptidx].pt.x << ", " << pts[ptidx].pt.y << ") - Harris response: "
                  << ((float)a * b - (float)c * c - harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq << "\n";
        pts[ptidx].response = ((float)a * b - (float)c * c -
                               harris_k * ((float)a + b) * ((float)a + b))*scale_sq_sq;
    }
}

void getHarrisScore(const cv::Mat &image, std::vector<Pixel> keypoints, int blockSize, int ksize, double k)
{
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_32F, 1, 0, ksize);
    cv::Sobel(image, Iy, CV_32F, 0, 1, ksize);
    float scale = (1 << (ksize - 1)) * 255.0 * blockSize;
    scale = 1.0f / scale;
    scale = scale * scale * scale * scale;

    for (auto &kp : keypoints)
    {
        int x = kp.getI();
        int y = kp.getJ();
        int halfBlock = blockSize / 2;

        float a = 0, b = 0, c = 0;

        for (int i = -halfBlock; i <= halfBlock; ++i)
        {
            for (int j = -halfBlock; j <= halfBlock; ++j)
            {
                int xi = x + i;
                int yj = y + j;

                float ix = Ix.at<float>(xi, yj);
                float iy = Iy.at<float>(xi, yj);

                a += ix * ix;
                b += iy * iy;
                c += ix * iy;

            }
        }
        float det = a * b - c * c;
        float trace = a + b;
        float R = ( det - k * trace * trace ) * scale;
        kp.setScore(R);
    }
}

int main() {
    // Load grayscale and color images
    cv::Mat img_gs = cv::imread("1.jpg", cv::IMREAD_GRAYSCALE);
    cv::Mat img = cv::imread("1.jpg", cv::IMREAD_COLOR);

    if (img.empty() || img_gs.empty()) {
        std::cout << "Could not read the image." << std::endl;
        return 1;
    }

    // Detect FAST keypoints
    std::vector<cv::KeyPoint> keypoints;
    std::vector<Pixel> keypixels;
    int fast_threshold = 20;
    bool nonmaxSuppression = true;
    // cv::FAST(img_gs, keypoints, fast_threshold, nonmaxSuppression);

    // Compute Harris responses for the detected keypoints
    int blockSize = 7; // Size of the neighborhood considered for Harris corner detection
    float harris_k = 0.04f; // Harris detector free parameter
    cv::KeyPoint max_kp(980, 671, 1);
    Pixel max_pixel(671, 980, img_gs.at<uchar>(671, 980), 0.0f);
    keypixels.push_back(max_pixel); // Add a pixel at (613, 483)
    keypoints.push_back(max_kp); // Add a keypoint at (613,
    HarrisResponses(img_gs, keypoints, blockSize, harris_k);
    getHarrisScore(img_gs, keypixels, blockSize, 3, 0.04);

    printf("Detected %zu keypoints\n", keypoints.size());
    std::sort(keypoints.begin(), keypoints.end(),
              [](const cv::KeyPoint& a, const cv::KeyPoint& b) {
                  return a.response < b.response;
              });
    // Draw detected corners
    for (const auto& kp : keypoints) {
        cv::circle(img, kp.pt, 3, cv::Scalar(0, 255, 0), -1);
        std::cout << "Keypoint at (" << kp.pt.x << ", " << kp.pt.y << ") with Harris response: " << kp.response << "\n";
    }

    for( const auto& kp : keypixels) {
        cv::circle(img, cv::Point(kp.getJ(), kp.getI()), 3, cv::Scalar(255, 0, 0), -1);
        std::cout << "Pixel at (" << kp.getJ() << ", " << kp.getI() << ") with Harris score: " << kp.getScore() << "\n";
    }

    cv::resize(img, img, cv::Size(), 0.5, 0.5);
    // Show result
    cv::imshow("FAST Corners with Harris Responses", img);
    cv::waitKey(0);

    return 0;
}