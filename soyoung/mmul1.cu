#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>

using namespace std;

__global__ void matrix_mul1(float *a, float *b, float *c, int m, int k, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
    if (col < k && row < m){
        for(int i = 0; i < n; i++){
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
}

int main(int argc, char* argv[])
{
    if (argc != 4) {
        return 0;
    }

    int m = strtol(argv[1], NULL, 10);
    int k = strtol(argv[2], NULL, 10);
    int n = strtol(argv[3], NULL, 10);
    printf("[Matrix multiplication, C = AB]\n");
    printf("\tA : (%d x %d) matrix, B : (%d x %d) matrix\n", m, k, k, n);
    printf("\tC : (%d x %d) matrix\n", m, n);

    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));

    if (h_A == NULL || h_B == NULL || h_C == NULL) {
        fprintf(stderr, "Failed to allocate host matrix!\n");
        exit(EXIT_SUCCESS);
    }
    
    // Matrix Initialization 
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            h_A[i*k + j] = rand() / (float)RAND_MAX;
        }
    }
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < n; j++) {
            h_B[i*n + j] = rand() / (float)RAND_MAX;
        }
    }

    // Device Matrix Allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(float));
    cudaMalloc((void**)&d_B, k * n * sizeof(float));
    cudaMalloc((void**)&d_C, m * n * sizeof(float));

    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice);

    GpuTimer timer;
    float time_mmul = 0.0;
    
    const int block_size = 16;
    dim3 threads(block_size, block_size);
    dim3 grid(ceil(m / (float)threads.x), ceil(n / (float)threads.y));
    printf("CUDA kernel launch with (%d x %d) blocks of (%d x %d) threads\n", grid.x, grid.y, threads.x, threads.y);

    cudaDeviceSynchronize();
    timer.Start();

    matrix_mul1<<<grid, threads>>>(d_A, d_B, d_C, m, k, n);

    cudaDeviceSynchronize();
    timer.Stop();

    time_mmul += timer.Elapsed();

    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("Elapsed Time: %fms\n", time_mmul);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");

    return 0;
}

