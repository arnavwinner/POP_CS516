#include <stdio.h>  
#include <cuda.h>
__global__   void dkernel(char *arr, int rrlen) { 	
	unsigned id = threadIdx.x;
	if (id < 5) {
		++arr[id];
	}
}
int main() {
	char cpuarr[] = "CS516",  *gpuarr;
	cudaMalloc(&gpuarr, sizeof(char) * (1 + strlen(cpuarr)));
	cudaMemcpy(gpuarr, cpuarr, sizeof(char) * (1 + strlen(cpuarr)), cudaMemcpyHostToDevice);
	dkernel<<<1, 32>>>(gpuarr, strlen(cpuarr));  cudaDeviceSynchronize();	// unnecessary.
	cudaMemcpy(cpuarr, gpuarr, sizeof(char) * (1 + strlen(cpuarr)), cudaMemcpyDeviceToHost);  
  printf(cpuarr);
	return 0;
}
