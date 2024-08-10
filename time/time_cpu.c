#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// 定义CPU计时器函数
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

// 假设这是您要测量的CUDA函数或内核
void function_to_measure(int *device_array, size_t size) {
    // 这里只是一个示例，实际代码中这里会是您的CUDA内核调用
    // 如：myKernel<<<gridSize, blockSize>>>(device_array, size);
}

int main() {
    int *device_array;
    size_t size = 1024; // 假设数组大小

    // 进行CUDA设备内存分配
    cudaMalloc(&device_array, size * sizeof(int));

    // 记录开始时间
    double iStart = cpuSecond();

    // 执行要测量的CUDA函数或内核
    function_to_measure(device_array, size);

    // 同步设备，确保测量包括所有GPU操作
    cudaDeviceSynchronize();

    // 记录结束时间并计算经过时间
    double iElaps = cpuSecond() - iStart;
    printf("Elapsed time: %f seconds\n", iElaps);

    // 清理：释放设备内存
    cudaFree(device_array);

    return 0;
}