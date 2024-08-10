
class GPUTimer
{
public:
    GPUTimer()
    {
        cudaEventCreate(&m_start);
        cudaEventCreate(&m_end);
    }

    ~GPUTimer()
    {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_end);
    }

    float elapsed_ms()
    {
        float ms = 0;
        cudaEventElapsedTime(&ms, m_start, m_end);
        return ms;
    }

    void start()
    {
        cudaEventRecord(m_start);
    }

    void stop()
    {
        cudaEventRecord(m_end);
        cudaEventSynchronize(m_end);
    }

private:
    cudaEvent_t m_start;
    cudaEvent_t m_end;
};

class CPUTimer
{
public:
    CPUTimer()
    {
        m_start = std::chrono::high_resolution_clock::now();
    }

    ~CPUTimer() {}

    void start()
    {
        m_start = std::chrono::high_resolution_clock::now();
    }

    void stop()
    {
        m_end = std::chrono::high_resolution_clock::now();
    }

    float elapsed_ms()
    {
        auto dur = std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start).count(); // us
        return (float)(dur) / 1000;
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_end;
};
