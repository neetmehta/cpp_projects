#include "orb.hpp"
#include "Descriptor.hpp"
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <string>

class Profiler
{
public:
    Profiler(const std::string &name)
        : name(name), start(std::chrono::high_resolution_clock::now()) {}

    ~Profiler()
    {
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
        // Profiler p("getHarrisScore");
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

    float getICAngle(cv::Mat &image, ORB::Keypoint &kp, int radius, std::vector<int> umax)
    {
        int m_01 = 0, m_10 = 0;
        int x0 = kp.getX();
        int y0 = kp.getY();

        for (int u = -radius; u <= radius; ++u)
        {
            m_10 += image.at<uchar>(y0, x0 + u) * u;
        }

        for (int v = 1; v <= radius; v++)
        {
            int d = umax.at(v);
            for (int u = -d; u <= d; u++)
            {

                m_10 += image.at<uchar>(y0 + v, x0 + u) * u;
                m_10 += image.at<uchar>(y0 - v, x0 + u) * u;

                m_01 += image.at<uchar>(y0 + v, x0 + u) * v;
                m_01 += image.at<uchar>(y0 - v, x0 + u) * (-v);
            }
        }

        return cv::fastAtan2((float)m_01, (float)m_10);
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

    void computeFastKeypoints(cv::Mat &image, std::vector<Keypoint> &keypoints, float fastThreshold, int HarrisblockSize, int nfeatures, int halfPatchSize, std::vector<int> umax, int level)
    {
        // Profiler p("computeFastKeypoints");
        std::vector<Keypoint> keypixels, keypixelsAfterNMS;

        int rows = image.rows;
        int cols = image.cols;

        for (int y = halfPatchSize; y < rows - halfPatchSize; ++y)
        {
            for (int x = halfPatchSize; x < cols - halfPatchSize; ++x)
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
                        centerPixel.setLevel(level);
                        centerPixel.setAngle(getICAngle(image, centerPixel, halfPatchSize, umax));
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

        int halfPatchSize = patchSize / 2;
        std::vector<int> umax(halfPatchSize + 2);

        int v, v0, vmax = cvFloor(halfPatchSize * sqrt(2.f) / 2 + 1);
        int vmin = cvCeil(halfPatchSize * sqrt(2.f) / 2);
        for (v = 0; v <= vmax; ++v)
            umax[v] = cvRound(sqrt((double)halfPatchSize * halfPatchSize - v * v));

        // Make sure we are symmetric
        for (v = halfPatchSize, v0 = 0; v >= vmin; --v)
        {
            while (umax[v0] == umax[v0 + 1])
                ++v0;
            umax[v] = v0;
            ++v0;
        }

        // Calculate patch extent

        for (int level = 0; level < nlevels; ++level)
        {
            std::vector<Keypoint> keypoints;
            computeFastKeypoints(imagePyramid[level], keypoints, fastThreshold, blockSize, nfeaturesPerLevel[level], halfPatchSize, umax, level);
            computeBriefDescriptor(imagePyramid[level], keypoints, 256, patchSize);
            for (auto &kp : keypoints)
            {
                // Scale keypoint coordinates to original image size
                float scale = std::pow(scaleFactor, level);
                kp.setX(cvRound(kp.getX() * scale));
                kp.setY(cvRound(kp.getY() * scale));
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

    // extern const int bit_pattern_31_[256][4];

    void computeBriefDescriptor(cv::Mat &image,
                                std::vector<Keypoint> &keypoints,
                                int nfeatures,     // usually 256
                                int patchSize)     // ORB default = 31
    {
        CV_Assert(patchSize == 31);  // current bit_pattern is for 31
        CV_Assert(nfeatures % 8 == 0);

        // Pre-blur the image (OpenCV does this at pyramid construction, not per kp)
        cv::Mat blurred;
        cv::GaussianBlur(image, blurred, cv::Size(7, 7), 2, 2, cv::BORDER_REFLECT_101);

        const int bytes = nfeatures / 8;

        for (auto &kp : keypoints)
        {
            // Allocate descriptor vector (zero initialized)
            std::vector<uchar> descriptor(bytes, 0);

            // Rotation (degrees → radians)
            float angle = kp.getAngle() * (float)(CV_PI / 180.f);
            float a = std::cos(angle);
            float b = std::sin(angle);

            int x0 = kp.getX();
            int y0 = kp.getY();

            for (int i = 0; i < nfeatures; i++)
            {
                int idx1 = bit_pattern_31_[i][0];
                int idy1 = bit_pattern_31_[i][1];
                int idx2 = bit_pattern_31_[i][2];
                int idy2 = bit_pattern_31_[i][3];

                // Rotate pattern points
                int r1 = cvRound(a * idx1 - b * idy1) + y0;
                int c1 = cvRound(b * idx1 + a * idy1) + x0;
                int r2 = cvRound(a * idx2 - b * idy2) + y0;
                int c2 = cvRound(b * idx2 + a * idy2) + x0;

                // Skip out-of-bounds
                if (r1 < 0 || r1 >= blurred.rows || c1 < 0 || c1 >= blurred.cols ||
                    r2 < 0 || r2 >= blurred.rows || c2 < 0 || c2 >= blurred.cols)
                {
                    continue;
                }

                uchar val1 = blurred.at<uchar>(r1, c1);
                uchar val2 = blurred.at<uchar>(r2, c2);

                // Pack bit (MSB-first like OpenCV)
                if (val1 < val2)
                    descriptor[i / 8] |= (1 << (7 - (i % 8)));
            }
            kp.setDescriptor(descriptor);
        }
    }



    std::vector<Keypoint> nonMaxSuppression(const std::vector<Keypoint> &keypoints, int radius, int rows, int cols)
    {
        // Profiler p("nonMaxSuppression");
        int cellSize = radius;
        int gridRows = (rows + cellSize - 1) / cellSize;
        int gridCols = (cols + cellSize - 1) / cellSize;

        std::vector<std::vector<Keypoint>> grid(gridRows * gridCols);

        // Assign keypoints to grid cells
        for (const auto &kp : keypoints)
        {
            int gx = kp.getX() / cellSize;
            int gy = kp.getY() / cellSize;
            grid[gy * gridCols + gx].push_back(kp);
        }

        // Pick strongest in each cell
        std::vector<Keypoint> result;

        for (auto &cell : grid)
        {
            if (cell.empty())
                continue;
            auto best = std::max_element(cell.begin(), cell.end(),
                                         [](const Keypoint &a, const Keypoint &b)
                                         {
                                             return a.getScore() < b.getScore();
                                         });
            result.push_back(*best);
        }

        return result;
    }

} // namespace vision