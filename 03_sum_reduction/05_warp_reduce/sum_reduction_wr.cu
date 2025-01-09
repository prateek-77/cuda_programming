#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

const int NUM_THREADS = 256;

// __device__ specifies that the function runs on the GPU, ans is called from the GPU.
// volatile tells the compiler not to perform optimization
// Optimizations such as caching values of shmem[tid] in a register
__device__ void warp_reduce(volatile int* shmem, const int tid) {
    // Each instruction below is executed in a lockstep (because all threads executing are in the same warp)
    // Warp execution in lockstep means that all threads in a warp execute the same instruction at the same time (SIMD)
    // Synchronization is implicit at every instruction
    shmem[tid] += shmem[tid + 32];
    shmem[tid] += shmem[tid + 16];
    shmem[tid] += shmem[tid + 8];
    shmem[tid] += shmem[tid + 4];
    shmem[tid] += shmem[tid + 2];
    shmem[tid] += shmem[tid + 1];

}

__global__ void sum_reduction(const int* arr, int* res_arr) {

    __shared__ int shmem[NUM_THREADS];
    
    int i = blockIdx.x*blockDim.x*2 + threadIdx.x;

    // Fill shared memory with elements
    shmem[threadIdx.x] = arr[i] + arr[i + blockDim.x];
    __syncthreads();

    // Loop with largest stride, and reduce it by 2 every iteration until it reaches 1.
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        int idx = threadIdx.x;
        if (idx < s) {
            shmem[idx] += shmem[idx + s];
        }
        __syncthreads();
    }

    // For the last 32 threads, since they belong to same warp,
    // we can perform loop unrolling to do warp execution
    // This eliminates the need for __syncthreads() which slows things down.
    if (threadIdx.x < 32) {
        warp_reduce(shmem, threadIdx.x);
    }

    if (threadIdx.x==0) res_arr[blockIdx.x] = shmem[0];
}


int main() {
    
    int N = 1 << 16;
    size_t bytes = sizeof(int) * N;
    size_t bytes_res = sizeof(int) * (N / NUM_THREADS); // 2^16 split by 256 threads = 256.

    vector<int> arr;
    vector<int> result_arr;

    arr.resize(N);
    result_arr.resize(N / NUM_THREADS);

    for (int i=0; i<N; i++) {
        arr[i] = rand() % 10;
    }

    int *arr_gpu, *result_arr_gpu;

    cudaMalloc(&arr_gpu, bytes);
    cudaMalloc(&result_arr_gpu, bytes_res);

    cudaMemcpy(arr_gpu, arr.data(), bytes, cudaMemcpyHostToDevice);

    int THREADS = NUM_THREADS;
    // Halve the number of blocks required
    int BLOCKS = (N + THREADS - 1) / THREADS / 2;

    // Reduce array from 2^16 to 2^8 (256)
    sum_reduction<<<BLOCKS, THREADS>>> (arr_gpu, result_arr_gpu);
    // Reduce above array to get result in the first index
    sum_reduction<<<1, THREADS>>> (result_arr_gpu, result_arr_gpu);

    cudaMemcpy(result_arr.data(), result_arr_gpu, bytes_res, cudaMemcpyDeviceToHost);
    assert(result_arr[0] == accumulate(arr.begin(), arr.end(), 0));

    cout << "Run Successful" << endl;

    cudaFree(arr_gpu);
    cudaFree(result_arr_gpu);
}