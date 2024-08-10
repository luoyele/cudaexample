
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <chrono>

// 定义全局变量
const int numElements = 5000000;

// vectorAdd 内核函数
__global__ void vectorAdd(const float *A, const float *B, float *C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

// warmup 内核函数
__global__ void warmup() {
    // 空内核，用于预热GPU
}

// CPU计时器函数
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int main(int argc, char **argv) {
    // 预热GPU
    warmup<<<1, 1>>>();
    cudaDeviceSynchronize();

    // 内存分配和初始化
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    size_t size = numElements * sizeof(float);

    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    for (int i = 0; i < numElements; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i);
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 计时方式选择
    double start = 0.0f, end = 0.0f;
    float elapsedTime = 0.0;
    cudaEvent_t event_start, event_stop;
    clock_t clock_start, clock_end;
    std::chrono::time_point<std::chrono::system_clock> c11_start, c11_end;

    if (argc < 2) {
        printf("Usage: %s <timing method>\n", argv[0]);
        return -1;
    }

    int timing_method = atoi(argv[1]);

    // 根据选择的计时方法进行计时
    if (timing_method == 1) {
        start = cpuSecond();
    } else if (timing_method == 2) {
        cudaEventCreate(&event_start);
        cudaEventCreate(&event_stop);
        cudaEventRecord(event_start, 0);
    } else if (timing_method == 3) {
        clock_start = clock();
    } else if (timing_method == 4) {
        c11_start = std::chrono::system_clock::now();
    }

    // 执行 vectorAdd 内核
    vectorAdd<<<(numElements + 255) / 256, 256>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize(); // 确保内核执行完成

    // 根据选择的计时方法结束计时
    if (timing_method == 1) {
        end = cpuSecond();
        printf("gettimeofday time = %lfms\n", (end - start) * 1000);
    } else if (timing_method == 2) {
        cudaEventRecord(event_stop, 0);
        cudaEventSynchronize(event_stop);
        cudaEventElapsedTime(&elapsedTime, event_start, event_stop);
        printf("cudaEvent time = %lfms\n", elapsedTime);
    } else if (timing_method == 3) {
        clock_end = clock();
        printf("clock time: %lfms.\n", ((double)(clock_end - clock_start) / CLOCKS_PER_SEC) * 1000);
    } else if (timing_method == 4) {
        c11_end = std::chrono::system_clock::now();
        auto elapsed_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(c11_end - c11_start).count();
        printf("chrono time: %dms.\n", elapsed_milliseconds);
    }

    // 内存释放
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // 清理cudaEvent
    if (timing_method == 2) {
        cudaEventDestroy(event_start);
        cudaEventDestroy(event_stop);
    }

    return 0;
}

// https://blog.csdn.net/litdaguang/article/details/77585011