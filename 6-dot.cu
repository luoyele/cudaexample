#include <stdio.h>
#include <cuda_runtime.h>

// 假设threadsPerBlock是线程块的大小，N是数组的长度
#define threadsPerBlock 256
#define N 1024


__global__ void dot(float *a, float *b, float *c, int N) {
    // 建立一个thread数量大小的共享内存数组
    extern __shared__ float cache[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;
    float temp = 0;

    // 计算点积
    while (tid < N) {
        temp += a[tid] * b[tid]; // 确保a和b的索引相同
        tid += blockDim.x * gridDim.x;
    }

    // 把算出的数存到cache里
    cache[cacheIndex] = temp;

    // 同步，确保所有线程都写入了共享内存
    __syncthreads();

    // 归约操作
    for (int i = blockDim.x / 2; i > 0; i >>= 1) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
    }

    // 每个块的第一个线程写入最终结果
    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

int main() {
    float *h_a, *h_b, *h_c; // 主机端数组
    float *d_a, *d_b, *d_c; // 设备端数组
    size_t size = N * sizeof(float); // 每个数组的大小

    // 分配主机内存
    h_a = (float *)malloc(size);
    h_b = (float *)malloc(size);
    h_c = (float *)malloc(size);

    // 初始化主机端数组
    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    // 分配设备内存
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // 从主机复制数据到设备
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // 计算线程块数量和每个线程块的线程数量
    int blockSize = threadsPerBlock;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // 调用内核函数
    dot<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // 从设备复制结果回主机
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // 验证结果（这里只是打印第一个结果，实际应用中可能需要更全面的验证）
    printf("Result of dot product: %f\n", h_c[0]);

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // 释放主机内存
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}