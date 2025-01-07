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

void verify_result(vector<int>& operand1, vector<int>& operand2, 
                   vector<int>& result, int N) {

    for (int i=0; i<N; i++) {
        assert(result[i] == operand1[i] + operand2[i]);
    }
}

int main() {

    // Vector has a size of 2^16 elements
    int N = 1 << 16;
    size_t bytes = sizeof(int) * N;

    vector<int> operand1;
    vector<int> operand2;
    vector<int> result;

    operand1.resize(N);
    operand2.resize(N);
    result.resize(N);

    for (int i=0; i<N; i++) {
        operand1[i] = rand() % 100;
        operand2[i] = rand() % 100;
    }

    int *operand1_gpu, *operand2_gpu, *result_gpu;
    
    // Allocate memory for variables on the GPU device
    cudaMalloc(&operand1_gpu, bytes);
    cudaMalloc(&operand2_gpu, bytes);
    cudaMalloc(&result_gpu, bytes);

    // Copy data from host to device
    // operand1.data() returns a raw pointer to the array of the vector, which is what cudaMemcpy requires.
    cudaMemcpy(operand1_gpu, operand1.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(operand2_gpu, operand2.data(), bytes, cudaMemcpyHostToDevice);

    // Define threads and blocks. For a thread block, max number of threads can be 1024
    int THREADS = 1 << 10;
    int BLOCKS = (N + THREADS - 1) / THREADS;

    vectorAdd<<<BLOCKS, THREADS>>>(operand1_gpu, operand2_gpu, result_gpu, N);

    cudaMemcpy(result.data(), result_gpu, bytes, cudaMemcpyDeviceToHost);

    verify_result(operand1, operand2, result, N);

    // Free memory on device
    cudaFree(operand1_gpu);
    cudaFree(operand2_gpu);
    cudaFree(result_gpu);

    cout << "Run Successful" << endl;

    return 0;
}