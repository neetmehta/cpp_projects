#include "orb.hpp"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string>

class Profiler {
public:
    Profiler(const std::string& name) 
        : name(name), start(std::chrono::high_resolution_clock::now()) {}

    ~Profiler() {
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[PROFILE] " << name << " took " << ms << " ms\n";
    }

private:
    std::string name;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
};

namespace ORB
{

    std::vector<Keypoint> FastPixels = {Keypoint(3, 0), Keypoint(-3, 0), Keypoint(0, 3), Keypoint(0, -3)};
    std::vector<Keypoint> surroundingPixels = {
        Keypoint(3, 0), Keypoint(3, 1), Keypoint(2, 2), Keypoint(1, 3), Keypoint(0, 3), Keypoint(-1, 3), Keypoint(-2, 2), Keypoint(-3, 1),
        Keypoint(-3, 0), Keypoint(-3, -1), Keypoint(-2, -2), Keypoint(-1, -3), Keypoint(0, -3), Keypoint(1, -3), Keypoint(2, -2), Keypoint(3, -1)};

    void getHarrisScore(const cv::Mat &image, std::vector<Keypoint> &keypoints, int blockSize, int ksize, double k)
    {
        Profiler p("getHarrisScore");
        cv::Mat Ix, Iy;
        cv::Sobel(image, Ix, CV_32F, 0, 1, ksize);
        cv::Sobel(image, Iy, CV_32F, 1, 0, ksize);
        float scale = (1 << (ksize - 1)) * 255.0 * blockSize;
        scale = 1.0f / scale;
        scale = scale * scale * scale * scale;

        for (auto &kp : keypoints)
        {
            int x0 = kp.getX();
            int y0 = kp.getY();
            int halfBlock = blockSize / 2;

            float a = 0, b = 0, c = 0;

            for (int i = -halfBlock; i <= halfBlock; ++i)
            {
                for (int j = -halfBlock; j <= halfBlock; ++j)
                {
                    int xi = x0 + i;
                    int yj = y0 + j;

                    float ix = Ix.at<float>(yj, xi);
                    float iy = Iy.at<float>(yj, xi);

                    a += ix * ix;
                    b += iy * iy;
                    c += ix * iy;
                }
            }
            float det = a * b - c * c;
            float trace = a + b;
            float R = (det - k * trace * trace) * scale;

            kp.setScore(R);
        }
    }

    void getICAngle(const cv::Mat &image, std::vector<Keypoint> &keypoints, std::vector<int> umax, int patchSize)
    {
        // Profiler p("getICAngle");
        int halfPatchSize = patchSize / 2;

        for (Keypoint &kp : keypoints)
        {
            float m01 = 0, m10 = 0;
            int x0 = kp.getX();
            int y0 = kp.getY();

            for (int v = -halfPatchSize; v < halfPatchSize; v++)
            {
                int u = umax[v];
                for (int x = -halfPatchSize; x > halfPatchSize; x++)
                {
                    if (x < std::abs(x))
                    {
                        float I = image.at<uchar>(v + x0, x + y0);
                        m10 += v * I;
                        m01 += x * I;
                    }
                }
            }
            kp.setAngle(std::atan2f(m01, m10));
        }
    }

    bool fastTest(cv::Mat &img, Keypoint p, float threshold)
    {
        int brighter = 0, darker = 0;
        uchar centerIntensity = static_cast<uchar>(p.getIntensity());

        for (const Keypoint &offset : FastPixels)
        {
            Keypoint fp = offset + p;
            uchar val = img.at<uchar>(fp.getY(), fp.getX());
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

    void buildImagePyramid(const cv::Mat &image, std::vector<cv::Mat> &imagePyramid,
                        int nlevels, float scaleFactor)
    {
        // Profiler p("buildImagePyramid");
        imagePyramid.clear();
        imagePyramid.resize(nlevels);
        imagePyramid[0] = image.clone();

        for (int level = 1; level < nlevels; ++level)
        {
            float scale = std::pow(scaleFactor, level);
            cv::Size sz(std::round((float)image.cols / scale),
                        std::round((float)image.rows / scale));

            cv::resize(image, imagePyramid[level], sz, 0, 0, cv::INTER_LINEAR);
        }
    }


    void computeFastKeypoints(cv::Mat &image, std::vector<Keypoint> &keypoints, float fastThreshold, int HarrisblockSize, int nfeatures)
    {
        // Profiler p("computeFastKeypoints");
        std::vector<Keypoint> keypixels, keypixelsAfterNMS;

        int rows = image.rows;
        int cols = image.cols;

        for (int y = 3; y < rows - 3; ++y)
        {
            for (int x = 3; x < cols - 3; ++x)
            {   
                if (!fastTest(image, Keypoint(x, y, image.at<uchar>(y, x), 0), fastThreshold))
                {
                    continue;
                }
                
                Keypoint centerPixel = Keypoint(x, y, image.at<uchar>(y, x), 0);
                int countBright = 0;
                int countDark = 0;

                // Check the 16 surrounding pixels

                for (int k = 0; k < 25; ++k)
                {
                    Keypoint surroundingPixel = surroundingPixels[k % 16] + centerPixel;
                    int xs = surroundingPixel.getX();
                    int ys = surroundingPixel.getY();
                    surroundingPixel.setIntensity(image.at<uchar>(ys, xs));

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

        keypixelsAfterNMS = nonMaxSuppression(keypixels, 7, image.rows, image.cols);

        getHarrisScore(image, keypixelsAfterNMS, HarrisblockSize, 3, 0.04);
        // Sort keypoints by Harris score in descending order

        std::sort(keypixelsAfterNMS.begin(), keypixelsAfterNMS.end(), [](const Keypoint &a, const Keypoint &b)
                  { return a.getScore() > b.getScore(); });

        for (size_t i = 0; i < keypixelsAfterNMS.size() && i < nfeatures; ++i)
        {
            keypoints.emplace_back(keypixelsAfterNMS[i]);
        }
    }

    std::vector<Keypoint> computeKeypoints(cv::Mat &image, float fastThreshold, int blockSize, int edgeThreshold, int patchSize, int nfeatures, int nlevels, float scaleFactor)
    {
        // Profiler p("computeKeypoints");
        // Initialize variables
        std::vector<int> nfeaturesPerLevel(nlevels);
        std::vector<cv::Mat> imagePyramid(nlevels);
        std::vector<Keypoint> allKeypoints;

        // Calculate the number of features per level
        float factor = (float)(1.0 / scaleFactor);
        float ndesiredFeaturesPerScale = nfeatures * (1 - factor) / (1 - (float)pow((double)factor, (double)nlevels));

        int sumFeatures = 0;
        for (int level = 0; level < nlevels - 1; level++)
        {
            nfeaturesPerLevel[level] = cvRound(ndesiredFeaturesPerScale);
            sumFeatures += nfeaturesPerLevel[level];
            ndesiredFeaturesPerScale *= factor;
        }
        nfeaturesPerLevel[nlevels - 1] = std::max(nfeatures - sumFeatures, 0);
        // Build image pyramid
        buildImagePyramid(image, imagePyramid, nlevels, scaleFactor);

        // Calculate patch extent
        
        for (int level = 0; level < nlevels; ++level)
        {
            std::vector<Keypoint> keypoints;
            computeFastKeypoints(imagePyramid[level], keypoints, fastThreshold, blockSize, nfeaturesPerLevel[level]);
            // Adjust keypoint coordinates to the original image scale
            float scale = std::pow(scaleFactor, level);
            for (auto &kp : keypoints)
            {
                kp.setLevel(level);
                allKeypoints.push_back(kp);
            }
        }

        // Sort all keypoints by score and retain the best nfeatures
        std::sort(allKeypoints.begin(), allKeypoints.end(), [](const Keypoint &a, const Keypoint &b)
                  { return a.getScore() > b.getScore(); });

        if (allKeypoints.size() > static_cast<size_t>(nfeatures))
        {
            allKeypoints.resize(nfeatures);
        }

        return allKeypoints;
    }

    std::vector<Keypoint> nonMaxSuppression(const std::vector<Keypoint>& keypoints, int radius, int rows, int cols)
    {
        // Profiler p("nonMaxSuppression");
        int cellSize = radius;
        int gridRows = (rows + cellSize - 1) / cellSize;
        int gridCols = (cols + cellSize - 1) / cellSize;

        std::vector<std::vector<Keypoint>> grid(gridRows * gridCols);

        // Assign keypoints to grid cells
        for (const auto& kp : keypoints) {
            int gx = kp.getX() / cellSize;
            int gy = kp.getY() / cellSize;
            grid[gy * gridCols + gx].push_back(kp);
        }

        // Pick strongest in each cell
        std::vector<Keypoint> result;
        
        for (auto& cell : grid) {
            if (cell.empty()) continue;
            auto best = std::max_element(cell.begin(), cell.end(),
                                        [](const Keypoint& a, const Keypoint& b) {
                                            return a.getScore() < b.getScore();
                                        });
            result.push_back(*best);
        }

        return result;
    }

} // namespace vision