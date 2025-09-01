#include "orb.hpp"
#include <cstdlib>

namespace ORB {

int getHarrisScore(const cv::Mat &image, Pixel p, int blockSize, int ksize, double k) {
    cv::Mat Ix, Iy;
    cv::Sobel(image, Ix, CV_32F, 1, 0, ksize);
    cv::Sobel(image, Iy, CV_32F, 0, 1, ksize);

    int x = p.getI();
    int y = p.getJ();

    int halfBlock = blockSize / 2;
    float A = 0.0f, B = 0.0f, C = 0.0f;

    for (int i = -halfBlock; i <= halfBlock; ++i) {
        for (int j = -halfBlock; j <= halfBlock; ++j) {
            if (x + i < 0 || x + i >= image.rows || y + j < 0 || y + j >= image.cols)
                continue;

            float ix = Ix.at<float>(x + i, y + j);
            float iy = Iy.at<float>(x + i, y + j);

            A += ix * ix;
            B += iy * iy;
            C += ix * iy;
        }
    }

    float detM = A * B - C * C;
    float traceM = A + B;

    return static_cast<int>(detM - k * traceM * traceM);
}

std::vector<Pixel> FastPixels = {Pixel(3,0,0,0), Pixel(-3,0,0,0), Pixel(0,3,0,0), Pixel(0,-3,0,0)};
std::vector<Pixel> surroundingPixels = {
    Pixel(3,0,0,0), Pixel(3,1,0,0), Pixel(2,2,0,0), Pixel(1,3,0,0), Pixel(0,3,0,0), Pixel(-1,3,0,0), Pixel(-2,2,0,0), Pixel(-3,1,0,0),
    Pixel(-3,0,0,0), Pixel(-3,-1,0,0), Pixel(-2,-2,0,0), Pixel(-1,-3,0,0), Pixel(0,-3,0,0), Pixel(1,-3,0,0), Pixel(2,-2,0,0), Pixel(3,-1,0,0)
};

bool fastTest(cv::Mat& img, Pixel p, float threshold) {
    int brighter = 0, darker = 0;
    uchar centerIntensity = static_cast<uchar>(p.getIntensity());

    for (const Pixel& offset : FastPixels) {
        Pixel fp = offset + p;
        uchar val = img.at<uchar>(fp.getI(), fp.getJ());
        if (val > centerIntensity + threshold) {
            brighter++;
        } else if (val < centerIntensity - threshold) {
            darker++;
        }
    }
    return (brighter > 1) || (darker > 1);
}

std::vector<std::pair<int, int>> computeFastKeypoints(cv::Mat& image, float threshold) {
    std::vector<Pixel> keypixels;

    int rows = image.rows;
    int cols = image.cols;

    for(int i=3; i<rows-3; ++i) {
        for(int j=3; j<cols-3; ++j) {
            if(!fastTest(image, Pixel(i, j, image.at<uchar>(i, j),0), threshold)) {
                continue;
            }

            Pixel centerPixel = Pixel(i, j, image.at<uchar>(i, j), 0);
            int countBright = 0;
            int countDark = 0;

            // Check the 16 surrounding pixels

            for(int k=0; k<25; ++k) {
                Pixel surroundingPixel = surroundingPixels[k%16] + centerPixel;
                int x = surroundingPixel.getI();
                int y = surroundingPixel.getJ();
                surroundingPixel.setIntensity(image.at<uchar>(x, y));

                // centerPixel.setScore(centerPixel.getScore() + std::abs(static_cast<int>(surroundingPixel.getIntensity()) - static_cast<int>(centerPixel.getIntensity())));

                if(surroundingPixel.getIntensity() > centerPixel.getIntensity() + threshold) {
                    countBright++;
                    countDark = 0;
                } else if(surroundingPixel.getIntensity() < centerPixel.getIntensity() - threshold) {
                    countDark++;
                    countBright = 0;
                } else {
                    countBright = 0;
                    countDark = 0;
                }

                if(countBright >= 9 || countDark >= 9) {
                    printf("Image size:");

                    centerPixel.setScore(getHarrisScore(image, centerPixel, 3, 3, 0.04));
                    keypixels.emplace_back(centerPixel);
                    break;
                }
            }
        }
    }

    // auto suppressed = nonMaxSuppression(keypixels, 2);
    // std::vector<std::pair<int, int>> keypoints;
    // for(size_t i = 0; i < keypixels.size(); ++i) {
    //     if(suppressed[i]==false) {
    //         keypoints.emplace_back(std::make_pair(keypixels[i].getI(), keypixels[i].getJ()));
    //     }
    // }

    std::sort(keypixels.begin(), keypixels.end(), [](const Pixel& a, const Pixel& b) {
        return a.getScore() > b.getScore();
    });
    std::vector<std::pair<int, int>> keypoints;
    for(size_t i = 0; i < keypixels.size() && i < 500; ++i) {
        keypoints.emplace_back(std::make_pair(keypixels[i].getI(), keypixels[i].getJ()));
    }
    return keypoints;
}

std::vector<bool> nonMaxSuppression(const std::vector<Pixel>& keypixels, int radius=3) {

    std::vector<bool> suppressed(keypixels.size(), false);

    for (size_t i = 0; i < keypixels.size(); ++i) {
        if (suppressed[i]) continue;

        const auto& kp1 = keypixels[i];
        for (size_t j = i + 1; j < keypixels.size(); ++j) {
            if (suppressed[j]) continue;

            const auto& kp2 = keypixels[j];
            int distSq = (kp1.getI() - kp2.getI()) * (kp1.getI() - kp2.getI()) +
                         (kp1.getJ() - kp2.getJ()) * (kp1.getJ() - kp2.getJ());
            if (distSq <= radius * radius) {
                if(kp1.getScore() >= kp2.getScore()) {
                    suppressed[j] = true;
                } else {
                    suppressed[i] = true;
                    break;
                }
            }
        }
    }
    return suppressed;
}

} // namespace vision