#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

#define WIDTH 16
#define TILE_WIDTH 2
#define N_THREADS 256

// Matrix multiplication kernel – per thread code
__global__ void MatrixMulKernel(int* d_M, int* d_N, int* d_P, int Width)
{
  // Calculate the row index of the d_P element and M
  int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  // Calculate the column index of d_P and N
  int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;
  float Pvalue = 0;
  // each thread computes one element of the block sub-matrix
  for (int k = 0; k < Width; ++k) Pvalue += d_M[Row*Width+k] * d_N[k*Width+Col];
  d_P[Row*Width+Col] = Pvalue;
}

float time_diff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

int main() {

    struct timeval start;
    struct timeval end;

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

    gettimeofday(&start, NULL);
    MatrixMulKernel<<<(WIDTH * WIDTH + N_THREADS - 1) / N_THREADS, N_THREADS>>>(d_M, d_N, d_P, WIDTH);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    cudaMemcpy(P, d_P, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix:");
    for (int j =0 ; j<WIDTH*WIDTH ; j=j+1){
        if (j%WIDTH == 0){printf("\n %d \t",*(P+j));}
        else{printf("%d\t",*(P+j));}
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

