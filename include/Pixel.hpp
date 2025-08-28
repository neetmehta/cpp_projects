struct Pixel
{
private:
    int i;     // Position
    int j;     // Position
    unsigned int I = 0; // Intensity
    unsigned int score = 0; // Score for corner strength

public:
    Pixel(int _i, int _j, unsigned int _I, unsigned int _score) : i(_i), j(_j), I(_I), score(_score) {}
    Pixel() = default;

    // Getters
    int getI() const { return i; }
    int getJ() const { return j; }
    unsigned int getIntensity() const { return I; }
    unsigned int getScore() const { return score; }

    // Setters
    void setI(int value) { i = value; }
    void setJ(int value) { j = value; }
    void setIntensity(unsigned int value) { I = value; }
    void setScore(unsigned int value) { score = value; }

    // Comparison operators
    bool operator<(const Pixel &other) const
    {
        return I < other.I;
    }
    bool operator>(const Pixel &other) const
    {
        return I > other.I;
    }
    bool operator<=(const Pixel &other) const
    {
        return I <= other.I;
    }
    bool operator>=(const Pixel &other) const
    {
        return I >= other.I;
    }
    Pixel operator+(const Pixel &other) const
    {
        return Pixel(i + other.i, j + other.j, I + other.I, score + other.score);
    }
    Pixel operator-(const Pixel &other) const
    {
        return Pixel(i - other.i, j - other.j, I - other.I, score - other.score);
    }
    bool operator==(const Pixel &other) const
    {
        return i == other.i && j == other.j && I == other.I && score == other.score;
    }
};