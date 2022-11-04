#include "Mat_Mul.cuh"
#include "../../DS_timer/DS_timer.h"

#define SIZE_M (512 * 2)
#define SIZE_N (512 * 4)
#define SIZE_K (512 * 2)

int main(int argc, char *argv[])
{
    // timer set
    DS_timer timer(4);
    timer.setTimerName(0, "[CPU]");
    timer.setTimerName(1, "[GPU]");
    timer.setTimerName(2, "[DATA Transfer] : Host->Device");
    timer.setTimerName(3, "[DATA Transfer] : Device->Host");

    // get matrix size spec
    // invalid argument, use default (1024_1024) x (1024_2048)
    int m, n, k;
    if (argc < 3) // default argument
    {
        m = SIZE_M;
        n = SIZE_N;
        k = SIZE_K;
    }
    else // argument user give
    {
        m = atoi(argv[1]);
        n = atoi(argv[2]);
        k = atoi(argv[3]);
    }

    printf("Step1: Size : A = (%d x %d), B = (%d x %d), C = (%d x %d)\n", m, k, k, n, m, n);

    int sizeA = m * k;
    int sizeB = k * n;
    int sizeC = m * n;

    // CPU matrix generation
    float *A = NULL float *B = NULL;
    allocNinitMem<float>(&A, sizeA);
    allocNinitMem<float>(&B, sizeB);

    float *h_C = NULL, float *d_C = NULL;
    allocNinitMem<float>(&h_C, sizeC);
    allocNinitMem<float>(&d_C, sizeC);

    for (int i = 0; i < sizeA; i++)
    {
        A[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }
    for (int i = 0; i < sizeB; i++)
    {
        B[i] = ((rand() % 10) + ((rand() % 100) / 100.0));
    }
    printf("Step2: CPU Matrix generation finished\n");

    // CPU MatMul
    timer.onTimer(0);
    for (int row = 0; row < m; row++)
    {
        for (int col = 0; col < n; col++)
        {
            int c_idx = ID2INDEX(row, col, n);
            h_c[c_idx] = 0;
            for (int j = 0; j < k; j++)
            {
                h_c[c_idx] += (A[ID2INDEX(row, i, k)] * B[ID2INDEX(i, col, n)]);
            }
        }
    }
    timer.offTimer(0);
    printf("Step3: CPU MatMul finished\n");

    // GPU matrix generation
    float *dA, *dB, *dC;

    cudaMalloc(&dA, sizeA * sizeof(float));
    cudaMalloc(&dB, sizeB * sizeof(float));
    cudaMalloc(&dC, sizeC * sizeof(float));

    cudaMemset(dA, 0, sizeA * sizeof(float));
    cudaMemset(dB, 0, sizeB * sizeof(float));
    cudaMemset(dC, 0, sizeC * sizeof(float));

    timer.onTimer(2);
    cudaMemcpy(dA, A, sizeA * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B, sizeB * sizeof(float), cudaMemcpyHostToDevice);
    timer.onTimer(2);

    printf("Step4: GPU matrix generation finished\n");

    // grid, block setting
    dim3 gridDim(ceil((float)m / BLOCK_SIZE), ceil((float)n / BLOCK_SIZE));
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    printf("Step6: Grid(%d, %d), Block(%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // GPU Matmul
    timer.onTimer(1);
    MatMul<<<gridDim, blockDim>>>(dA, dB, dC, m, n, k);
    cudaDeviceSynchronize();
    timer.offTimer(1);
    printf("Step7: GPU matrix multiplication finished\n");

    timer.onTimer(3);
    cudaMemcpy(d_C, dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost);
    timer.offTimer(3);
    pritnf("Step8: GPU result transfer to CPU finished\n");

    bool result = true;
    for (int i = 0; i < sizeC; i++)
    {
        if (d_C[i] != dC[i])
        {
            printf("[%d] not matched! (%f, %f)\n", i, d_C[i], dC[i]);
            result = false;
        }
    }
    if (result)
    {
        pritnf("GPU work well!\n");
    }
    return result;
}
