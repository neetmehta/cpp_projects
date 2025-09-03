#include "orb.hpp"
#include <cstdlib>

namespace ORB
{

    void getHarrisScore(const cv::Mat &image, std::vector<Pixel> &keypoints, int blockSize, int ksize, double k)
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
            float R = (det - k * trace * trace) * scale;
            // printf("Pixel at (%d, %d) - Harris score: %f\n", x, y, R);
            kp.setScore(R);
        }
    }

    std::vector<Pixel> FastPixels = {Pixel(3, 0), Pixel(-3, 0), Pixel(0, 3), Pixel(0, -3)};
    std::vector<Pixel> surroundingPixels = {
        Pixel(3, 0), Pixel(3, 1), Pixel(2, 2), Pixel(1, 3), Pixel(0, 3), Pixel(-1, 3), Pixel(-2, 2), Pixel(-3, 1),
        Pixel(-3, 0), Pixel(-3, -1), Pixel(-2, -2), Pixel(-1, -3), Pixel(0, -3), Pixel(1, -3), Pixel(2, -2), Pixel(3, -1)};

    bool fastTest(cv::Mat &img, Pixel p, float threshold)
    {
        int brighter = 0, darker = 0;
        uchar centerIntensity = static_cast<uchar>(p.getIntensity());

        for (const Pixel &offset : FastPixels)
        {
            Pixel fp = offset + p;
            uchar val = img.at<uchar>(fp.getI(), fp.getJ());
            if (val > centerIntensity + threshold)
            {
                brighter++;
            }
            else if (val < centerIntensity - threshold)
            {
                darker++;
            }
        }
        return (brighter > 1) || (darker > 1);
    }

    std::vector<std::pair<int, int>> computeORBKeypoints(cv::Mat &image, float fastThreshold, int blockSize, int edgeThreshold, int patchSize, int nfeatures, int nlevels, float scaleFactor)
    {
        std::vector<cv::Mat> imagePyramid(nlevels);
        std::vector<int> nfeaturesPerLevel(nlevels);
        std::vector<Pixel> keypixels, keypixelsAfterNMS;
        imagePyramid[0] = image.clone();
        for (int level = 1; level < nlevels; ++level)
        {
            float scale = std::pow(scaleFactor, level);
            cv::resize(image, imagePyramid[level],
                       cv::Size(std::round(image.cols / scale),
                                std::round(image.rows / scale)),
                       0, 0, cv::INTER_LINEAR);
        }
        
        float factor = (float)(1.0 / scaleFactor);
        float ndesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));
        int sumFeatures = 0;
        for( int level = 0; level < nlevels-1; level++ )
        {
            nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
            sumFeatures += nfeaturesPerLevel[level];
            ndesiredFeaturesPerScale *= factor;
        }

        int rows = image.rows;
        int cols = image.cols;

        int halfBlock = blockSize / 2;
        int borderOffset = halfBlock + 1;

        for (int i = 3; i < rows - 3; ++i)
        {
            for (int j = 3; j < cols - 3; ++j)
            {
                if (!fastTest(image, Pixel(i, j, image.at<uchar>(i, j), 0), fastThreshold))
                {
                    continue;
                }

                Pixel centerPixel = Pixel(i, j, image.at<uchar>(i, j), 0);
                int countBright = 0;
                int countDark = 0;

                // Check the 16 surrounding pixels

                for (int k = 0; k < 25; ++k)
                {
                    Pixel surroundingPixel = surroundingPixels[k % 16] + centerPixel;
                    int x = surroundingPixel.getI();
                    int y = surroundingPixel.getJ();
                    surroundingPixel.setIntensity(image.at<uchar>(x, y));

                    if (surroundingPixel.getIntensity() > centerPixel.getIntensity() + fastThreshold)
                    {
                        countBright++;
                        countDark = 0;
                    }
                    else if (surroundingPixel.getIntensity() < centerPixel.getIntensity() - fastThreshold)
                    {
                        countDark++;
                        countBright = 0;
                    }
                    else
                    {
                        countBright = 0;
                        countDark = 0;
                    }

                    if (countBright >= 9 || countDark >= 9)
                    {
                        keypixels.emplace_back(centerPixel);
                        break;
                    }
                }
            }
        }

        auto suppressed = nonMaxSuppression(keypixels, 7);
        for (size_t i = 0; i < keypixels.size(); ++i)
        {
            if (!suppressed[i])
            {
                keypixelsAfterNMS.emplace_back(keypixels[i]);
            }
        }

        getHarrisScore(image, keypixelsAfterNMS, blockSize, 3, 0.04);
        // Sort keypoints by Harris score in descending order

        std::sort(keypixels.begin(), keypixels.end(), [](const Pixel &a, const Pixel &b)
                  { return a.getScore() > b.getScore(); });
        std::vector<std::pair<int, int>> keypoints;
        for (size_t i = 0; i < keypixelsAfterNMS.size() && i < nfeatures; ++i)
        {
            keypoints.emplace_back(std::make_pair(keypixelsAfterNMS[i].getI(), keypixelsAfterNMS[i].getJ()));
            printf("Keypoint at (%d, %d) with score: %f\n", keypixelsAfterNMS[i].getI(), keypixelsAfterNMS[i].getJ(), keypixelsAfterNMS[i].getScore());
        }
        return keypoints;
    }

    std::vector<std::pair<int, int>> computeORBKeypointsPyramid(cv::Mat &image,
                                                                float fastThreshold,
                                                                int blockSize,
                                                                int edgeThreshold,
                                                                int patchSize,
                                                                int nfeatures,
                                                                int nlevels,
                                                                float scaleFactor)
    {
        std::vector<Pixel> allKeypixels;

        for (int level = 0; level < nlevels; ++level)
        {
            float scale = std::pow(scaleFactor, level);
            cv::Mat scaledImg;
            cv::resize(image, scaledImg,
                       cv::Size(std::round(image.cols / scale),
                                std::round(image.rows / scale)),
                       0, 0, cv::INTER_LINEAR);

            // Detect keypoints at this pyramid level
            auto keypointsAtLevel = computeORBKeypoints(scaledImg,
                                                        fastThreshold,
                                                        blockSize,
                                                        edgeThreshold,
                                                        patchSize,
                                                        nfeatures);

            // Convert back to Pixel with proper coordinates rescaled
            for (auto &kp : keypointsAtLevel)
            {
                int origY = std::round(kp.first * scale);
                int origX = std::round(kp.second * scale);
                allKeypixels.emplace_back(Pixel(origY, origX,
                                                image.at<uchar>(origY, origX), 0));
            }
        }

        // Compute Harris scores on the **original image** for all rescaled pixels
        getHarrisScore(image, allKeypixels, blockSize, 3, 0.04);

        // Sort globally
        std::sort(allKeypixels.begin(), allKeypixels.end(),
                  [](const Pixel &a, const Pixel &b)
                  {
                      return a.getScore() > b.getScore();
                  });

        // Keep top N
        std::vector<std::pair<int, int>> finalKeypoints;
        for (size_t i = 0; i < allKeypixels.size() && i < (size_t)nfeatures; ++i)
        {
            finalKeypoints.emplace_back(
                std::make_pair(allKeypixels[i].getI(),
                               allKeypixels[i].getJ()));
        }

        return finalKeypoints;
    }

    std::vector<bool> nonMaxSuppression(const std::vector<Pixel> &keypixels, int radius = 3)
    {

        std::vector<bool> suppressed(keypixels.size(), false);

        for (size_t i = 0; i < keypixels.size(); ++i)
        {
            if (suppressed[i])
                continue;

            const auto &kp1 = keypixels[i];
            for (size_t j = i + 1; j < keypixels.size(); ++j)
            {
                if (suppressed[j])
                    continue;

                const auto &kp2 = keypixels[j];
                int distSq = (kp1.getI() - kp2.getI()) * (kp1.getI() - kp2.getI()) +
                             (kp1.getJ() - kp2.getJ()) * (kp1.getJ() - kp2.getJ());
                if (distSq <= radius * radius)
                {
                    if (kp1.getScore() >= kp2.getScore())
                    {
                        suppressed[j] = true;
                    }
                    else
                    {
                        suppressed[i] = true;
                        break;
                    }
                }
            }
        }
        return suppressed;
    }

} // namespace vision