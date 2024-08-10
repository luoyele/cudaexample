#include <cuda_runtime.h>
#include <iostream>

int main() {
    cudaEvent_t start, stop;
    float elapsedTime;

    // 创建事件
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(start, 0);

    // 执行内核函数（这里用一个假设的函数名代替）
    myKernel<<<gridSize, blockSize>>>(...);

    // 记录结束时间
    cudaEventRecord(stop, 0);

    // 同步事件，确保结束时间已经记录
    cudaEventSynchronize(stop);

    // 计算经过时间
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Elapsed Time: " << elapsedTime << " ms" << std::endl;

    // 销毁事件
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}