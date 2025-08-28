#include "keypoints.hpp"
#include <cstdlib>

namespace vision {

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

            for(int k=0; k<16; ++k) {
                Pixel surroundingPixel = surroundingPixels[k] + centerPixel;
                int x = surroundingPixel.getI();
                int y = surroundingPixel.getJ();
                surroundingPixel.setIntensity(image.at<uchar>(x, y));

                centerPixel.setScore(centerPixel.getScore() + std::abs(static_cast<int>(surroundingPixel.getIntensity()) - static_cast<int>(centerPixel.getIntensity())));

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
                    keypixels.emplace_back(centerPixel);
                    break;
                }
            }
        }
    }
    auto suppressed = nonMaxSuppression(keypixels, 2);
    std::vector<std::pair<int, int>> keypoints;
    for(size_t i = 0; i < keypixels.size(); ++i) {
        if(suppressed[i]==false) {
            keypoints.emplace_back(std::make_pair(keypixels[i].getI(), keypixels[i].getJ()));
        }
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