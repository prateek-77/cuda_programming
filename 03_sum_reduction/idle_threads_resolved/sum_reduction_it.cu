#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

const int NUM_THREADS = 256;

__global__ void sum_reduction(const int* arr, int* res_arr) {

    __shared__ int shmem[NUM_THREADS];
    
    // Start performing sum reduction while filling shared memory itself.
    // This reduces the number of blocks required by half!
    // Number of threads per block still remain the same.
    // Overall, number of idle threads consumed by the program is still reduced.
    int i = blockIdx.x*blockDim.x*2 + threadIdx.x;

    // Fill shared memory with elements
    shmem[threadIdx.x] = arr[i] + arr[i + blockDim.x];
    __syncthreads();

    // Loop with largest stride, and reduce it by 2 every iteration until it reaches 1.
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        int idx = threadIdx.x;
        if (idx < s) {
            shmem[idx] += shmem[idx + s];
        }
        __syncthreads();
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