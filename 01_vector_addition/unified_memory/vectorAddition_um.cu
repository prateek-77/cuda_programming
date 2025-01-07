#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// CUDA kernel
// __global__ keyword indicates a kernel function that will
// run on the GPU and is callable from the host (CPU)
__global__ void vectorAdd(const int* operand1, const int* operand2, 
                          int* result, const int N) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N) {
        result[tid] = operand1[tid] + operand2[tid];
    }
}

void verify_result(const int* operand1, const int* operand2, 
                   const int* result, int N) {

    for (int i=0; i<N; i++) {
        assert(result[i] == operand1[i] + operand2[i]);
    }
}

int main() {

    // Vector has a size of 2^16 elements
    int N = 1 << 16;
    size_t bytes = sizeof(int) * N;

    int* operand1;
    int* operand2;
    int* result;


    // Here, we use cudaMallocManaged
    // It allows automatic management of memory by the Unified Memory System
    // Memory allocated using this is automatically accessible by both host and device
    // No need for explicit copying between host and device
    // Slower than original approach due to overheads
    cudaMallocManaged(&operand1, bytes);
    cudaMallocManaged(&operand2, bytes);
    cudaMallocManaged(&result, bytes);

    for (int i=0; i<N; i++) {
        operand1[i] = rand() % 100;
        operand2[i] = rand() % 100;
    }

    // Define threads and blocks. For a thread block, max number of threads can be 1024
    int THREADS = 1 << 10;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    vectorAdd<<<BLOCKS, THREADS>>>(operand1, operand2, result, N);
    
    // Earlier, CudaMemcpy acted as an implicit synchronization mechanism
    // Here, due to its absence, we need to invoke it explicitly
    // to allow all threads to finish execution
    cudaDeviceSynchronize();

    verify_result(operand1, operand2, result, N);

    // Free memory on device
    cudaFree(operand1);
    cudaFree(operand2);
    cudaFree(result);

    cout << "Run Successful" << endl;

    return 0;
}