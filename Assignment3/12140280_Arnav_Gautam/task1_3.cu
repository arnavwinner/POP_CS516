#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define WIDTH 16
#define TILE_WIDTH 2
#define N_THREADS 256

__global__ void matrixMulTiled(const int* M, const int* N, int* P, int width) {
    __shared__ int Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ int Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    int Pvalue = 0;

    for (int ph = 0; ph < width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[row * width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * width + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    P[row * width + col] = Pvalue;
}

float time_diff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

int main() {
    int* M, *N, *P;
    int size = WIDTH * WIDTH * sizeof(int);

    M = (int*)malloc(size);
    N = (int*)malloc(size);
    P = (int*)malloc(size);

    for (int i = 0; i < WIDTH * WIDTH; ++i) {
        M[i] = rand() % 10;
        N[i] = rand() % 10;
    }

    int *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size);
    cudaMalloc((void**)&d_N, size);
    cudaMalloc((void**)&d_P, size);

    cudaMemcpy(d_M, M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, size, cudaMemcpyHostToDevice);

    dim3 dimGrid(WIDTH / TILE_WIDTH, WIDTH / TILE_WIDTH);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

    struct timeval start, end;
    gettimeofday(&start, NULL);
    matrixMulTiled<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, WIDTH);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix:\n");
    for (int j = 0; j < WIDTH * WIDTH; ++j) {
        if (j % WIDTH == 0) {
            printf("\n %d \t", P[j]);
        } else {
            printf("%d\t", P[j]);
        }
    }
    printf("\n");
    printf("time spent: %0.8f sec\n", time_diff(&start, &end));
    

    free(M);
    free(N);
    free(P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    return 0;
}
