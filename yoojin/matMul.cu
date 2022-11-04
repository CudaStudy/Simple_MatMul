 #include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define M 32
#define K 16
#define N 64

__global__ void vecMul(int *_a, int *_b, int *_c, int M, int K, int N){
	int row = blockIdx.y*blockDim.y + threadIdx.y;
	int col = blockIdx.x*blockDim.x + threadIdx.x;

	int tmpSum = 0;
	for(int i=0; i<K; i++){
		tmpSum += _a[row*K + i] * _b[i*N + col];
	}
	_c[row*N + col] = tmpSum;
} 

int main(void){
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;

	int MK = M * K; int KN = K * N; int MN = M * N;

	int AMemSize = sizeof(int) * MK;
	int BMemSize = sizeof(int) * KN;
	int CMemSize = sizeof(int) * MN;

	a = new int[MK]; memset(a, 0, AMemSize);
	b = new int[KN]; memset(b, 0, BMemSize);
	c = new int[MN]; memset(c, 0, CMemSize);

	for(int i=0; i<MK; i++){
		a[i] = rand();
	}
	for(int i=0; i<KN; i++){
		b[i] = rand();
	}

	cudaMalloc(&d_a, AMemSize);
	cudaMalloc(&d_b, BMemSize);
	cudaMalloc(&d_c, CMemSize);

	cudaMemcpy(d_a, a, AMemSize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, BMemSize, cudaMemcpyHostToDevice);
	
	dim3 dimGrid(N/32, M/32, 1);
	dim3 dimBlock(32, 32, 1);
	vecMul<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, M, K, N);
	
	cudaDeviceSynchronize();
	cudaMemcpy(c, d_c, memSize, cudaMemcpyDeviceToHost);
	
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
	delete [] a; delete [] b; delete [] c;

	return 0;	
}



