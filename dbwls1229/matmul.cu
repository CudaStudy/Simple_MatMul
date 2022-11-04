#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#define M 1024
#define N 2048
#define K 1024

#define A_SIZE (M*K)
#define B_SIZE (K*N)
#define C_SIZE (M*N)

#define THREADS_X 32
#define THREADS_Y 32

/* KERNEL FUNCTION */
__global__ void matMul(float *_A, float *_B, float *_C){
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = N * row + col;

    if (row < M && col < N){
        float val = 0.0;
        for (int k = 0; k < K; k++){
            val += _A[row * K + k] * _B[k * N + col];
        }
        _C[idx] = val;
    }
}

/* MAIN FUNCTION */
int main(void){
    /*
    float A[M][K];
    float B[K][N];
    float hostC[M][N];
    float deviceC[M][N];
    */

    float *dA, *dB, *dC;
    dA = dB = dC = NULL;

    dim3 dimGrid(ceil((float)N / THREADS_X), ceil((float)M / THREADS_Y), 1);
    dim3 dimBlock(THREADS_X, THREADS_Y, 1);

    struct timeval startTime, endTime;
    double elapsedTime = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0;

    int numThreads = THREADS_X * THREADS_Y;
    int numBlocks = ceil((float)M / THREADS_X) * ceil((float)N / THREADS_Y);
    int numData = C_SIZE;
    int numOps = numData > (numBlocks * numThreads) ? (numData / (numBlocks * numThreads)) + (numData % (numBlocks * numThreads) > 0 ? 1 : 0): 1;
    printf("Need %d threads\n", numData);
    printf("numOps : %d\n", numOps);

    // Host memory allocation
    float* A = new float[A_SIZE];
    float* B = new float[B_SIZE];
    float* hostC = new float[C_SIZE];
    float* deviceC = new float[C_SIZE];

    for (int i = 0; i < M; i++){
        for (int j = 0; j < K; j++){
            A[i*K+j] = rand() % 100;
        }
    }
    for (int i = 0; i < K; i++){
        for (int j = 0; j < N; j++){
            B[i*N+j] = rand() % 100;
        }
    }   

    // Device memory allocation
    cudaMalloc(&dA, sizeof(float)*A_SIZE);
    cudaMalloc(&dB, sizeof(float)*B_SIZE);
    cudaMalloc(&dC, sizeof(float)*C_SIZE);

    // Copy input data from host to device
    cudaMemcpy(dA, A, sizeof(float)*A_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeof(float)*B_SIZE, cudaMemcpyHostToDevice);
    
    // Kernel call (GPU computation)
    cudaEventRecord(start, 0);
    matMul<<<dimGrid, dimBlock>>>(dA, dB, dC);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy output data from device to host
    cudaMemcpy(deviceC, dC, sizeof(float)*C_SIZE, cudaMemcpyDeviceToHost);

    // CPU computation
    gettimeofday(&startTime, NULL);
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            hostC[i*N+j] = 0.0;
            for (int k = 0; k < K; k++){
                hostC[i*N+j] += A[i*K+k] * B[k*N+j];
            }
        }
    }
    gettimeofday(&endTime, NULL);
    elapsedTime += (endTime.tv_sec - startTime.tv_sec) * 1000. + (endTime.tv_usec - startTime.tv_usec) / 1000.;  // in ms

    // Check results
    bool result = true;
    for (int i = 0; i < M; i++){
        for (int j = 0; j < N; j++){
            if (hostC[i*N+j] != deviceC[i*N+j]){
                printf("[%d]th result is not matched! (CPU : %f vs GPU : %f)\n", i*N+j, hostC[i*N+j], deviceC[i*N+j]);
                result = false;                
            }
        }
    }

    if (result){
        printf("GPU works well-!\n");
    }

    printf("-------------------------------------------------\n");
    printf("# of blocks : %d, # of threads : %d\n", numBlocks, numThreads);
    printf("-------------------------------------------------\n");
    printf("CPU execution time : %fms\n", elapsedTime);
    printf("GPU execution time : %fms\n", time);
    printf("-------------------------------------------------\n");
    printf("GPU speedup over CPU : %f\n", elapsedTime / time);

    // Host & Device memory release
    delete [] A;
    delete [] B;
    delete [] hostC;
    delete [] deviceC;
 
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}
