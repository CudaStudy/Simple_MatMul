#include "Mat_Mul.cuh"

__global__ void MatMul(float *matA, float *matB, float *matC, int m, int n, int k)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    int col = blockDim.y * blockIdx.y + threadIdx.y;

    if (m <= row || n <= col)
    {
        return; // finished job
    }

    float val = 0; // register
    for (int i = 0; i < k; i++)
    {
        val += __fmul_rn(matA[ID2INDEX(row, i, k)], matB[ID2INDEX(i, col, n)]);
    }
    matC[ID2INDEX(row, col, n)] = val;
}

template <class T>
void allocNinitMem(T **p, long long size, double *memUsage = NULL);