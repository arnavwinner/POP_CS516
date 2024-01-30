%%cu
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define K 128

__global__ void fun(int *a, int *b, int *c) {
	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

int main() {
	// we first allocate memory to CPU
	printf("Hello World!");
	int a[K * K], b[K * K], c[K * K], i;
 	for (i = 0; i < K * K; i++) {
	    a[i] = i;
	    b[i] = i;
	}
 	printf("%d\n", a[0]);
	// now make GPU Device Pointers
	int *d_a, *d_b, *d_c;
	// allocate memory to GPU
	cudaMalloc((void**)&d_a, K * K * sizeof(int));
	cudaMalloc((void**)&d_b, K * K * sizeof(int));
	cudaMalloc((void**)&d_c, K * K * sizeof(int));
	// now we copy from CPU to GPU
	cudaMemcpy(d_a, a, K * K * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, K * K * sizeof(int), cudaMemcpyHostToDevice);
	// now do the calculations, call the kernel
	fun<<<1, K * K>>>(d_a, d_b, d_c);
	cudaMemcpy(c, d_c, K * K * sizeof(int), cudaMemcpyDeviceToHost); // we are now ready with the sum matrix
	for (i = 0; i < K * K; ++i)
		printf("%d\n", c[i]);
 // free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
	return 0;
}

