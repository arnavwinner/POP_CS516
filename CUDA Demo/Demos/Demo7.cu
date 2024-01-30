#include <stdio.h>  
#include <cuda.h>  
#define N 100

__global__ void fun(int *a) {
	a[threadIdx.x] = threadIdx.x * threadIdx.x;
}

int main() {
	int a[N], *da;  int i;
	cudaMalloc(&da, N * sizeof(int));
	fun<<<1, N>>>(da);
	cudaMemcpy(a, da, N * sizeof(int),
	cudaMemcpyDeviceToHost);  
	for (i = 0; i < N; ++i)
		printf("%d\n", a[i]);  
	return 0;
}

