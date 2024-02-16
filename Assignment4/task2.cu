%%cuda

#include <stdio.h>
#include <cuda_runtime.h>

#define N 100000000 // Size of the array

// Array of Structures (AOS) layout
typedef struct {
    float x;
    float y;
    float z;
} ParticleAOS;

// Structure of Arrays (SOA) layout
typedef struct {
    float *x;
    float *y;
    float *z;
} ParticleSOA;

__global__ void sumAOS(ParticleAOS *particles, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (tid < N) {
        sum = particles[tid].x + particles[tid].y + particles[tid].z;
    }
    atomicAdd(result, sum);
}

__global__ void sumSOA(ParticleSOA particles, float *result) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    if (tid < N) {
        sum = particles.x[tid] + particles.y[tid] + particles.z[tid];
    }
    atomicAdd(result, sum);
}

int main() {
    // Allocate memory for AOS layout
    ParticleAOS *h_particlesAOS;
    cudaMallocHost((void **)&h_particlesAOS, N * sizeof(ParticleAOS));

    // Initialize data for AOS layout
    for (int i = 0; i < N; ++i) {
        h_particlesAOS[i].x = 1.0f;
        h_particlesAOS[i].y = 2.0f;
        h_particlesAOS[i].z = 3.0f;
    }

    // Allocate memory for SOA layout
    ParticleSOA h_particlesSOA;
    cudaMallocHost((void **)&h_particlesSOA.x, N * sizeof(float));
    cudaMallocHost((void **)&h_particlesSOA.y, N * sizeof(float));
    cudaMallocHost((void **)&h_particlesSOA.z, N * sizeof(float));

    // Initialize data for SOA layout
    for (int i = 0; i < N; ++i) {
        h_particlesSOA.x[i] = 1.0f;
        h_particlesSOA.y[i] = 2.0f;
        h_particlesSOA.z[i] = 3.0f;
    }

    // Allocate memory for result
    float *d_resultAOS, *d_resultSOA;
    cudaMalloc((void **)&d_resultAOS, sizeof(float));
    cudaMalloc((void **)&d_resultSOA, sizeof(float));
    cudaMemset(d_resultAOS, 0, sizeof(float));
    cudaMemset(d_resultSOA, 0, sizeof(float));

    // Copy data to device
    ParticleAOS *d_particlesAOS;
    cudaMalloc((void **)&d_particlesAOS, N * sizeof(ParticleAOS));
    cudaMemcpy(d_particlesAOS, h_particlesAOS, N * sizeof(ParticleAOS), cudaMemcpyHostToDevice);

    ParticleSOA d_particlesSOA;
    cudaMalloc((void **)&d_particlesSOA.x, N * sizeof(float));
    cudaMalloc((void **)&d_particlesSOA.y, N * sizeof(float));
    cudaMalloc((void **)&d_particlesSOA.z, N * sizeof(float));
    cudaMemcpy(d_particlesSOA.x, h_particlesSOA.x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particlesSOA.y, h_particlesSOA.y, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particlesSOA.z, h_particlesSOA.z, N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Run kernel for AOS layout
    sumAOS<<<gridSize, blockSize>>>(d_particlesAOS, d_resultAOS);

    // Run kernel for SOA layout
    sumSOA<<<gridSize, blockSize>>>(d_particlesSOA, d_resultSOA);

    // Copy result from device to host
    float h_resultAOS, h_resultSOA;
    cudaMemcpy(&h_resultAOS, d_resultAOS, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_resultSOA, d_resultSOA, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Result for AOS layout: %.2f\n", h_resultAOS);
    printf("Result for SOA layout: %.2f\n", h_resultSOA);

    // Free memory
    cudaFreeHost(h_particlesAOS);
    cudaFreeHost(h_particlesSOA.x);
    cudaFreeHost(h_particlesSOA.y);
    cudaFreeHost(h_particlesSOA.z);
    cudaFree(d_particlesAOS);
    cudaFree(d_particlesSOA.x);
    cudaFree(d_particlesSOA.y);
    cudaFree(d_particlesSOA.z);
    cudaFree(d_resultAOS);
    cudaFree(d_resultSOA);

    return 0;
}
