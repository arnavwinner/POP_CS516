#include <stdio.h>  
#include <cuda.h>
__global__ void dkernel() {  
    printf("Hello World.\n");
}
int main() {
    dkernel<<<1, 32>>>();  
    cudaDeviceSynchronize();  
    return 0;
}

