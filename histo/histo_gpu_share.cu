#include <iostream>

#define DATA_LEN (100 * 1024 * 1024)

inline int rnd(float x)
{
    return static_cast<int>(x * rand() / RAND_MAX);
}

inline void check(cudaError_t call, const char* file, const int line)
{
    if (call != cudaSuccess)
    {
        std::cout << "cuda error: " << cudaGetErrorName(call) << std::endl;
        std::cout << "error file: " << file << " error line: " << line << std::endl;
        std::cout << cudaGetErrorString(call) << std::endl;
    }
}

#define CHECK(call) (check(call, __FILE__, __LINE__))


__global__ void cal_hist(unsigned char* buffer, unsigned int* hist, long data_size)
{
    // init
    __shared__ unsigned int temp[256];
    temp[threadIdx.x] = 0;
    __syncthreads();

    int index = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    while (index < data_size)
    {
        atomicAdd(&temp[buffer[index]], 1);
        index += stride;
    }
    __syncthreads();

    atomicAdd(&(hist[threadIdx.x]), temp[threadIdx.x]);
}

int main(void)
{
    unsigned char* buffer = new unsigned char[DATA_LEN];
    for (int i = 0; i < DATA_LEN; ++i)
    {
        buffer[i] = rnd(255);
        if (buffer[i] > 255)
        {
            std::cout << "error" << std::endl;
        }
    }
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    CHECK(cudaEventRecord(start, 0));

    unsigned int* d_hist;
    CHECK(cudaMalloc((void**)&d_hist, sizeof(unsigned int) * 256));
    // new function
    CHECK(cudaMemset(d_hist, 0, sizeof(int)));

    unsigned char* d_buffer;
    CHECK(cudaMalloc((void**)&d_buffer, sizeof(unsigned char) * DATA_LEN));
    CHECK(cudaMemcpy(d_buffer, buffer, sizeof(unsigned char) * DATA_LEN, cudaMemcpyHostToDevice));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    int block_num = prop.multiProcessorCount;
    cal_hist<<<block_num, 256>>>(d_buffer, d_hist, DATA_LEN);

    unsigned int h_hist[256];
    CHECK(cudaMemcpy(h_hist, d_hist, sizeof(unsigned int) * 256, cudaMemcpyDeviceToHost));
    CHECK(cudaEventRecord(end, 0));
    CHECK(cudaEventSynchronize(end));
    float elapsed_time;
    CHECK(cudaEventElapsedTime(&elapsed_time, start, end));
    std::cout << "using time: " << elapsed_time << "ms" << std::endl;

    long hist_count = 0;
    for (int i = 0; i <256; ++i)
    {
        hist_count += h_hist[i];
    }
    std::cout << "histogram sum: " << hist_count << std::endl;

    for (int i = 0; i < DATA_LEN; ++i)
    {
        h_hist[buffer[i]]--;
    }
    for (int i = 0; i < 256; ++i)
    {
        if (h_hist[i] != 0)
        {
            std::cout << "cal error" << std::endl;
        }
    }
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(end));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_buffer));

    delete[] buffer;

    return 0;
}

// https://blog.csdn.net/weixin_45773137/article/details/125584905