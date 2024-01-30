#include <stdio.h>

#define THREADS_PER_BLOCK 16

// CUDA kernel for matrix addition
__global__ void matrixAddition(int* M, int* N, int* result, int k) {
    // Calculate the global indices for the current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Perform element-wise addition
    if (row < k && col < k) {
        result[row * k + col] = M[row * k + col] + N[row * k + col];
    }
}

int main() {
    const int k = 16; // Adjust the matrix size (k x k) as needed

    // Allocate memory for matrices M, N, and the result on the host
    int *h_M, *h_N, *h_result;
    h_M = (int*)malloc(k * k * sizeof(int));
    h_N = (int*)malloc(k * k * sizeof(int));
    h_result = (int*)malloc(k * k * sizeof(int));

    // Initialize matrices M and N on the host
    for (int i = 0; i < k * k; ++i) {
        h_M[i] = i;
        h_N[i] = i * 2;
    }

    // Allocate memory for matrices M, N, and the result on the device
    int *d_M, *d_N, *d_result;
    cudaMalloc((void**)&d_M, k * k * sizeof(int));
    cudaMalloc((void**)&d_N, k * k * sizeof(int));
    cudaMalloc((void**)&d_result, k * k * sizeof(int));

    // Copy matrices M and N from host to device
    cudaMemcpy(d_M, h_M, k * k * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, h_N, k * k * sizeof(int), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks((k + threadsPerBlock.x - 1) / threadsPerBlock.x, (k + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixAddition<<<numBlocks, threadsPerBlock>>>(d_M, d_N, d_result, k);

    // Copy the result matrix from device to host
    cudaMemcpy(h_result, d_result, k * k * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result matrix
    printf("Matrix M:\n");
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%d ", h_M[i * k + j]);
        }
        printf("\n");
    }

    printf("\nMatrix N:\n");
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%d ", h_N[i * k + j]);
        }
        printf("\n");
    }

    printf("\nResult Matrix M+N:\n");
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j < k; ++j) {
            printf("%d ", h_result[i * k + j]);
        }
        printf("\n");
    }

    // Free allocated memory on the host and device
    free(h_M);
    free(h_N);
    free(h_result);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_result);

    return 0;
}
