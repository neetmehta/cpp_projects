#pragma once
#include <vector>
#include <opencv2/opencv.hpp>
namespace ORB
{
    struct Keypoint
    {
    private:
        int x;           // Position
        int y;           // Position
        float I = 0.0f;  // Intensity
        float score = 0.0f; // Score for corner strength
        float angle = 0.0f; // Angle for orientation 
        int level = 0; // Pyramid level
        std::vector<uchar> descriptor; // Descriptor (256 bits = 32 bytes)

    public:
        Keypoint(int _x, int _y, float _I, float _score, float _angle, int _level) : x(_x), y(_y), I(_I), score(_score), angle(angle), level(_level) {}    
        Keypoint(int _x, int _y, float _I, float _score, float _angle) : x(_x), y(_y), I(_I), score(_score), angle(angle) {}
        Keypoint(int _x, int _y, float _I, float _score) : x(_x), y(_y), I(_I), score(_score) {}
        Keypoint(int _x, int _y) : x(_x), y(_y) {}
        Keypoint() = default;

        // Getters
        int getX() const { return x; }
        int getY() const { return y; }
        float getIntensity() const { return I; }
        float getScore() const { return score; }
        float getAngle() const { return angle; }
        int getLevel() const { return level; }
        const std::vector<uchar> &getDescriptor() const { return descriptor; }

        // Setters
        void setX(int value) { x = value; }
        void setY(int value) { y = value; }
        void setIntensity(float value) { I = value; }
        void setScore(float value) { score = value; }
        void setLevel(int l) { level = l; }
        void setAngle(float a) { angle = a; }
        void setDescriptor(const std::vector<uchar> &desc) { descriptor = desc; }

        // Comparison operators
        bool operator<(const Keypoint &other) const
        {
            return I < other.I;
        }
        bool operator>(const Keypoint &other) const
        {
            return I > other.I;
        }
        bool operator<=(const Keypoint &other) const
        {
            return I <= other.I;
        }
        bool operator>=(const Keypoint &other) const
        {
            return I >= other.I;
        }
        Keypoint operator+(const Keypoint &other) const
        {
            return Keypoint(x + other.x, y + other.y, I + other.I, score + other.score);
        }
        Keypoint operator-(const Keypoint &other) const
        {
            return Keypoint(x - other.x, y - other.y, I - other.I, score - other.score);
        }
        bool operator==(const Keypoint &other) const
        {
            return x == other.x && y == other.y && I == other.I && score == other.score && angle == other.angle && level == other.level;
        }
    };

}
