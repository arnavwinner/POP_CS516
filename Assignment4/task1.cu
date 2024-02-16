%%cuda

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16 // Define the tile size for matrix multiplication

__global__ void matrixVectorMulWithoutCoalescing(float *M, float *V, float *Z, int m) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < m) {
        float sum = 0.0f;
        for (int i = 0; i < m; ++i) {
            sum += V[i] * M[i * m + tid];
        }
        Z[tid] = sum;
    }
}

__global__ void matrixVectorMulWithCoalescing(float *M, float *V, float *Z, int m) {
    __shared__ float s_V[TILE_SIZE];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;

    if (tid < m) {
        float sum = 0.0f;
        s_V[idx] = V[tid];
        __syncthreads();
        for (int i = 0; i < m; ++i) {
            sum += s_V[i] * M[i * m + tid];
        }
        Z[tid] = sum;
    }
}

void printMatrix(float *matrix, int m) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            printf("%.2f ", matrix[i * m + j]);
        }
        printf("\n");
    }
}

int main() {
    int m = 32; // size of the matrix and vector (reduced for readability)
    int size = m * m * sizeof(float); // size of matrix M
    int vSize = m * sizeof(float); // size of vector V

    // Allocate memory for the host
    float *h_M = (float *)malloc(size);
    float *h_V = (float *)malloc(vSize);
    float *h_Z = (float *)malloc(vSize);

    // Initialize matrix M and vector V
    for (int i = 0; i < m * m; ++i) {
        h_M[i] = 1.0f; // fill matrix M with 1s for simplicity
    }
    for (int i = 0; i < m; ++i) {
        h_V[i] = 1.0f; // fill vector V with 1s for simplicity
    }

    // Allocate memory for the device
    float *d_M, *d_V, *d_Z;
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_V, vSize);
    cudaMalloc((void **)&d_Z, vSize);

    // Copy data from host to device
    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;

    // Perform matrix-vector multiplication without coalesced memory accesses
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixVectorMulWithoutCoalescing<<<gridSize, blockSize>>>(d_M, d_V, d_Z, m);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time for matrix-vector multiplication without coalesced memory accesses: %f milliseconds\n", milliseconds);

    // Copy the result back to the host
    cudaMemcpy(h_Z, d_Z, vSize, cudaMemcpyDeviceToHost);

    // Print the result matrix
    // printf("Result Matrix without coalesced memory accesses:\n");
    // printMatrix(h_Z, m);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_V);
    cudaFree(d_Z);

    // Perform matrix-vector multiplication with coalesced memory accesses
    cudaMalloc((void **)&d_M, size);
    cudaMalloc((void **)&d_V, vSize);
    cudaMalloc((void **)&d_Z, vSize);

    cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, vSize, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    
    matrixVectorMulWithCoalescing<<<gridSize, blockSize>>>(d_M, d_V, d_Z, m);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("\nTime for matrix-vector multiplication with coalesced memory accesses: %f milliseconds\n", milliseconds);

    cudaMemcpy(h_Z, d_Z, vSize, cudaMemcpyDeviceToHost);

    // Print the result matrix
    // printf("Result Matrix with coalesced memory accesses:\n");
    // printMatrix(h_Z, m);

    // Free host memory
    free(h_M);
    free(h_V);
    free(h_Z);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_V);
    cudaFree(d_Z);

    return 0;
}
