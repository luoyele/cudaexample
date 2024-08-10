#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)                                                     \
    do                                                                  \
    {                                                                   \
        cudaError_t ret = call;                                         \
        if (ret != cudaSuccess)                                         \
        {                                                               \
            printf("        CUDA Error\n");                             \
            printf("        File:%s\n", __FILE__);                      \
            printf("        Line:%d\n", __LINE__);                      \
            printf("        Error code:%d\n", ret);                     \
            printf("        Error text:%s\n", cudaGetErrorString(ret)); \
        }                                                               \
    } while (0)


int main()
{
    int device_count = 0;
    CHECK(cudaGetDeviceCount(&device_count));
    if(device_count > 0) {
        std::cout << "Found " << device_count << " GPUs!" << std::endl;
        for(int id=0;id<device_count;id++) {
            std::cout << "Device: " << id << std::endl;
            cudaDeviceProp prop;
            CHECK(cudaGetDeviceProperties(&prop, id));

            printf("GPU Name: %s\n", prop.name);
            printf("GPU Global Memory(显存容量): %f GB\n", (float)prop.totalGlobalMem/(1024*1024*1024));
            printf("GPU Memory 位宽:%d bit\n", prop.memoryBusWidth);
            printf("GPU SM个数:%d\n", prop.multiProcessorCount);
            printf("GPU 每个SM上最大线程数量:%d\n", prop.maxThreadsPerMultiProcessor);
        }

    }else {
        std::cout << "No NVIDIA GPU Exist !" << std::endl;
    }
}