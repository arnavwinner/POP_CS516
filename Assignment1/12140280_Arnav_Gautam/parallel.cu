// author: Arnav Gautam

#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

// CUDA kernel to compute the array B in parallel
__global__ void computeArrayB(int* A, int* B, int N, int R) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        int sm = 0;
        for (int r = 0; r < R; ++r) {
            if (r == 0) {
                sm += A[idx];
                continue;
            }
            if (idx - r >= 0) {
                sm += A[idx - r];
            }
            if (idx + r < N) {
                sm += A[idx + r];
            }
        }
        B[idx] = sm;
    }
}

int main() {
    int N, R;
    N = 131072;
    R = 32;

    int* A = new int[N];
    for (int i = 0; i < N; ++i) A[i] = i;

    int* B = new int[N];
    int* d_A, *d_B;

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));

    // Copy input data from host to device
    cudaMemcpy(d_A, A, N * sizeof(int), cudaMemcpyHostToDevice);

    // Define block size and grid size
    int blockSize = 1024;
    int gridSize = 128;

    auto start1 = high_resolution_clock::now();

    // Launch the CUDA kernel
    computeArrayB<<<gridSize, blockSize>>>(d_A, d_B, N, R);

    // Copy the result back to the host
    cudaMemcpy(B, d_B, N * sizeof(int), cudaMemcpyDeviceToHost);

    auto stop1 = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop1 - start1);
    cout << "Time: " << duration.count() / 1e6 << endl;

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);

    // Free host memory
    delete[] A;
    delete[] B;

    return 0;
}
