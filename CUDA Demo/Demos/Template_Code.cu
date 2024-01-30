%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024

// CUDA kernel to perform the array computation
_global_ void computeArray(int a, int* X, int* Y, int* Z) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        Z[tid] = a * X[tid] + Y[tid];
    }
}

int main() {
    // Initialize data on the host

    int a = 2; // set value of "a" as you may want.

    // Allocate x, y and z in the host.
    int X[N], Y[N], Z[N];

    for (int i = 0; i < N; ++i) {
        X[i] = i;
        Y[i] = N - i;
    }                         // we are just taking sample values for arrays: X and Y;

    // Declare GPU memory pointers... we call it device_x, device_y, device_z
    int *d_X, *d_Y, *d_Z;

    // CUDA Malloc for d_x , d_y, d_z in the device (GPU)
    cudaMalloc((void**)&d_X, N * sizeof(int));
    cudaMalloc((void**)&d_Y, N * sizeof(int));
    cudaMalloc((void**)&d_Z, N * sizeof(int));

    //Copy X and Y from host to GPU Device  
    cudaMemcpy(d_X, X, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Y, Y, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the kernel
    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    computeArray<<<grid_size, block_size>>>(a, d_X, d_Y, d_Z);

    // Copy d_z from  GPU Device to CPU.
    cudaMemcpy(Z, d_Z, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup --> cleans up the GPU memory that we have used.
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(d_Z);

    // Print the result
    for (int i = 0; i < N; ++i) {
        printf("Z[%d] = %d\n", i, Z[i]);
    }

    return 0;
}

