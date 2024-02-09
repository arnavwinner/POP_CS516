#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <cuda.h>

const int N = 4194304;
const int threadsPerBlock = 256;
const int arraySize = N;
const int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;

__global__ void reduce0(int *g_idata, int *g_odata) {
  extern __shared__ int sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  __syncthreads();
  for (unsigned int s=blockDim.x/2; s>0; s>>=1) 
  {
    if (tid < s) {
    sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }
  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

float time_diff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

int main() {
    struct timeval start, end;

    int* h_input = (int*)malloc(arraySize * sizeof(int));
    int* h_output = (int*)malloc(blocksPerGrid * sizeof(int));


    for (int i = 0; i < arraySize; ++i) {
        h_input[i] = i;
    }

    int* d_input, *d_output;
    cudaMalloc((void**)&d_input, arraySize * sizeof(int));
    cudaMalloc((void**)&d_output, blocksPerGrid * sizeof(int));

    cudaMemcpy(d_input, h_input, arraySize * sizeof(int), cudaMemcpyHostToDevice);

    
    gettimeofday(&start, NULL);

    reduce0<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);
    cudaDeviceSynchronize();

    gettimeofday(&end, NULL);

    cudaMemcpy(h_output, d_output, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

    int result = 0;
    for (int i = 0; i < blocksPerGrid; ++i) {
        result += h_output[i];
    }

    printf("time spent: %0.8f sec\n", time_diff(&start, &end));

    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}