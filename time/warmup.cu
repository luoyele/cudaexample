#include <cuda_runtime.h>

// 定义一个简单的CUDA内核，用于warmup
__global__ void warmupKernel() {
    // 这里可以是一些简单的操作，比如线程ID的计算
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // 执行一些计算，以触发GPU预热
}

// 定义warmup函数
void gpuWarmup() {
    int warmupIterations = 100; // 预热迭代次数

    for (int i = 0; i < warmupIterations; ++i) {
        warmupKernel<<<1, 256>>>();
        cudaDeviceSynchronize(); // 确保每次迭代后GPU都完全同步
    }
}

int main() {
    // 在性能测量之前调用warmup函数
    gpuWarmup();

    // ... 接下来是性能测量代码 ...

    return 0;
}