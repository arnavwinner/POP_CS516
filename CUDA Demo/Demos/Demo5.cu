#include <stdio.h>
#include <cuda.h>  
#define N 100
__global__    void fun() {
    printf("%d\n", threadIdx.x*threadIdx.x);
}
int main() {
    fun<<<1, N>>>();
    cudaDeviceSynchronize();  
    return 0;
}

