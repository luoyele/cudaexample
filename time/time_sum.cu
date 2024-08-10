#include<iostream>
#include<algorithm>
#include<cmath>
#include<cuda_runtime.h>

void elmtwise_sum_cpu(int* arr1, int* arr2, int* out, int N) {
    for(int i=0;i<N;i++) out[i] = arr1[i] + arr2[i];
}

__global__ void kernel_sum(int* arr1, int* arr2, int* out, int N) {
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(thread_id < N) {
        out[thread_id] = arr1[thread_id] + arr2[thread_id];
    }
}

void elmtwise_sum_gpu(int* arr1, int* arr2, int* out, int N) {
    // 1. GPU端申请显存
    int* d_arr1 = nullptr;
    int* d_arr2 = nullptr;
    int* d_out = nullptr;
    cudaMalloc((void**)&d_arr1, sizeof(int)*N);
    cudaMalloc((void**)&d_arr2, sizeof(int)*N);
    cudaMalloc((void**)&d_out, sizeof(int)*N);

    // 2. CPU Memory数据复制到GPU显存
    cudaMemcpy(d_arr1, arr1, sizeof(int)*N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr2, arr2, sizeof(int)*N, cudaMemcpyHostToDevice);

    // 3. 设置GPU端线程配置, launch the GPU kernel
    int blk_size = 128;
    int grid_size = (N + blk_size -1) / blk_size;
    kernel_sum<<<grid_size, blk_size>>>(d_arr1, d_arr2, d_out, N);

    // 4. Cpoy GPU result to CPU
    cudaMemcpy(out, d_out, sizeof(int)*N, cudaMemcpyDeviceToHost);

    // 5. Free GPU Memory
    cudaFree(d_arr1);
    cudaFree(d_arr2);
    cudaFree(d_out);
}


int main() {

    const int N = 512* 512;
    int* arr1 = new int[N];
    int* arr2 = new int[N];
    int* out_cpu = new int[N];
    int* out_gpu = new int[N];
    srand(123456);
    for(int i=0;i<N;i++) {
        arr1[i] = rand() * 5 % 255;
        arr2[i] = rand() % 128 + 5;
    }

    
    elmtwise_sum_cpu(arr1, arr2, out_cpu, N);
    elmtwise_sum_gpu(arr1, arr2, out_gpu, N);

    auto print_array = [](int* arr, int N, int k, const std::string& msg) -> void {
        std::cout << msg << std::endl;
        int n = std::min(N, k);
        for(int i=0;i<n;i++) std::cout << arr[i] <<" ";
        std::cout << std::endl;
    };

    print_array(out_cpu, N, 10, "CPU");
    print_array(out_gpu, N, 10, "GPU");

    // validate
    int i=0;
    for(i=0;i<N;i++){
        if(out_cpu[i] != out_gpu[i]){
            std::cout << "Error, not equal!" << std::endl;
            break;
        }
    }

    if(i==N) std::cout << "Test OK, all correct !" << std::endl;

    delete[] arr1;
    delete[] arr2;
    delete[] out_cpu;
    delete[] out_gpu;

    return 0;
}

// https://www.jianshu.com/p/f5e7ff72f421