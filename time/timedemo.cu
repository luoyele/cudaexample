#include<iostream>
#include<cuda_runtime.h>

// ref: https://blog.csdn.net/qq_17239003/article/details/78991567

using namespace std;

#define CHECK(call)                                                         \
    do                                                                      \
    {                                                                       \
        const cudaError_t error_code = call;                                \
        if (error_code != cudaSuccess)                                      \
        {                                                                   \
            printf("CUDA Error\n");                                         \
            printf("    File:   %s\n", __FILE__);                           \
            printf("    Line:   %d\n", __LINE__);                           \
            printf("    Error code: %d\n", error_code);                     \
            printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
            exit(1);                                                        \
        }                                                                   \
    } while (0)


// GPU Kernel func, perform element-wise add
__global__ void kernel_sum(int *arr1, int *arr2, int *out, int N)
{
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id < N)
    {
        out[thread_id] = arr1[thread_id] + arr2[thread_id];
    }
}

int main() {
    const int N = 2048 * 2048;
    int *arr1 = new int[N];
    int *arr2 = new int[N];
    int *out = new int[N];
    srand(123456);
    for (int i = 0; i < N; i++)
    {
        arr1[i] = rand() * 5 % 255;
        arr2[i] = rand() % 128 + 5;
    }

    // 1. GPU端申请显存
    int *d_arr1 = nullptr;
    int *d_arr2 = nullptr;
    int *d_out = nullptr;
    CHECK(cudaMalloc((void **)&d_arr1, sizeof(int) * N));
    CHECK(cudaMalloc((void **)&d_arr2, sizeof(int) * N));
    CHECK(cudaMalloc((void **)&d_out, sizeof(int) * N));

    // 2. CPU Memory数据复制到GPU显存
 
    cudaMemcpy(d_arr1, arr1, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, sizeof(int) * N, cudaMemcpyHostToDevice);

    // 3. 设置GPU端线程执行配置, launch the GPU kernel
    cudaEvent_t start, end;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&end));
    // 开始计时
    cudaEventRecord(start);

    int blk_size = 128;
    int grid_size = (N + blk_size - 1) / blk_size;
    kernel_sum<<<grid_size, blk_size>>>(d_arr1, d_arr2, d_out, N);

    // 结束计时
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    // 统计时间
    float time_ms = 0.f;
    cudaEventElapsedTime(&time_ms, start, end);
    std::cout << "CUDA Kernel time: " << time_ms << " ms" << std::endl;

    
    // 4. Cpoy GPU result to CPU
    cudaMemcpy(out, d_out, sizeof(int) * N, cudaMemcpyDeviceToHost);
   
    // 5. Free GPU Memory
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_out);
}