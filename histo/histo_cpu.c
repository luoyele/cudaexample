#include <iostream>
#include <chrono>

#define DATA_LEN (100 * 1024 * 1024)

inline int rnd(float x)
{
    return static_cast<int>(x * rand() / RAND_MAX);
}

int main(void)
{
    // cpu hist
    unsigned char* buffer = new unsigned char[DATA_LEN];
    for (int i = 0; i < DATA_LEN; ++i)
    {
        buffer[i] = rnd(255);
        if (buffer[i] > 255)
        {
            std::cout << "error" << std::endl;
        }
    }
    int hist[256];
    for (int i = 0; i < 256; ++i)
    {
        hist[i] = 0;
    }

    auto start = std::chrono::system_clock::now();
    for (int i = 0; i < DATA_LEN; ++i)
    {
        hist[buffer[i]]++;
    }
    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-start);
    std::cout << double(duration.count()) * std::chrono::microseconds::period::num / std::chrono::microseconds::period::den << "s" << std::endl;

    long hist_count = 0;
    for (int i = 0; i < 256; ++i)
    {
        hist_count += hist[i];
    }
    if (hist_count != DATA_LEN)
    {
        std::cout << "error" << std::endl;
    }

    delete[] buffer;

    return 0;
}