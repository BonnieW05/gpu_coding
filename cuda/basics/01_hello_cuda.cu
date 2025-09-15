/*
CUDA基础 - Hello CUDA
学习CUDA编程的基本概念和第一个CUDA程序
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA内核函数
__global__ void hello_cuda() {
    // 获取线程ID
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello CUDA from thread %d!\n", tid);
}

int main() {
    printf("=== Hello CUDA 程序 ===\n");
    
    // 设置网格和块的大小
    int num_blocks = 2;
    int threads_per_block = 4;
    
    // 启动CUDA内核
    hello_cuda<<<num_blocks, threads_per_block>>>();
    
    // 等待GPU完成
    cudaDeviceSynchronize();
    
    printf("CUDA内核执行完成！\n");
    
    return 0;
}

/*
编译命令:
nvcc -o hello_cuda 01_hello_cuda.cu

运行命令:
./hello_cuda
*/
