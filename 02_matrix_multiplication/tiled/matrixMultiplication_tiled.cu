#include <stdlib.h>
#include <assert.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

__global__ void matrixMultiplication(const int* matrixA, const int* matrixB,
                                     int* matrixC, int N) {

    // CUDA shared memory does not allow dynamic memory allocation
    __shared__ int sharedA[256];
    __shared__ int sharedB[256];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;  

    int tmp = 0;

    // Loop over all shared memory tiles
    for (int k=0; k<N; k+=blockDim.x) {

        // Fill the shared memory tile
        sharedA[threadIdx.y*blockDim.x + threadIdx.x] = matrixA[row*N + k + threadIdx.x];
        sharedB[threadIdx.y*blockDim.x + threadIdx.x] = matrixB[(k + threadIdx.y)*N + col];

        // Let all threads in the block finish filling the shared memory tile
        // Acts like block level synchronization barrier
        __syncthreads();

        // Perform matmul operation using shared memory tiles
        for (int idx=0; idx<blockDim.x; idx++) {
            tmp += sharedA[threadIdx.y*blockDim.x + idx] * 
                                    sharedB[idx*blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    matrixC[row*N + col] = tmp;
                    
}

void verify_result(const int* matrixA, const int* matrixB,
                                     int* matrixC, int N) {

    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
            int curr_val = 0;
            for (int k=0; k<N; k++) {
                curr_val += matrixA[i*N + k] * matrixB[k*N + j];
            }
            assert(matrixC[i*N + j] == curr_val);
        }
    }
}

int main() {

    // 1024 rows and columns
    int N = 1 << 10;
    size_t bytes = sizeof(int) * N * N;

    int* matrixA = new int[N*N];
    int* matrixB = new int[N*N];
    int* matrixC = new int[N*N];

    for (int i=0; i<N*N; i++) {
        matrixA[i] = rand() % 100;
        matrixB[i] = rand() % 100;
        matrixC[i] = 0;
    }

    int *matrixA_gpu, *matrixB_gpu, *matrixC_gpu;

    cudaMalloc(&matrixA_gpu, bytes);
    cudaMalloc(&matrixB_gpu, bytes);
    cudaMalloc(&matrixC_gpu, bytes);

    cudaMemcpy(matrixA_gpu, matrixA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_gpu, matrixB, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixC_gpu, matrixC, bytes, cudaMemcpyHostToDevice);

    int THREADS = 16;
    int BLOCKS = (N + THREADS - 1) / THREADS;
    
    dim3 block_size (THREADS, THREADS);
    dim3 grid_size (BLOCKS, BLOCKS);

    matrixMultiplication<<<grid_size, block_size>>>(matrixA_gpu, matrixB_gpu, matrixC_gpu, N);

    cudaMemcpy(matrixC, matrixC_gpu, bytes, cudaMemcpyDeviceToHost);

    verify_result(matrixA, matrixB, matrixC, N);

    cudaFree(matrixA_gpu);
    cudaFree(matrixB_gpu);
    cudaFree(matrixC_gpu);

    cout << "Run Successful" << endl;

}