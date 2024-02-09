#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define WIDTH 16


// Matrix multiplication on the (CPU) host
void MatrixMulOnHost(int* M, int* N, int* P, int Width)
{
  for (int i = 0; i < Width; ++i)
  {
    for (int j = 0; j < Width; ++j) 
    {
      double sum = 0;
      for (int k = 0; k < Width; ++k) 
      {
      double a = M[i * Width + k];
      double b = N[k * Width + j];
      sum += a * b;
      }
      P[i * Width + j] = sum;
    }
  }
}

float time_diff(struct timeval *start, struct timeval *end) {
  return (end->tv_sec - start->tv_sec) + 1e-6 * (end->tv_usec - start->tv_usec);
}

int main() {
    struct timeval start, end;


    int* M, *N, *P;
    int* a;
    int size = WIDTH * WIDTH * sizeof(int);


    M = (int*)malloc(size);
    N = (int*)malloc(size);
    P = (int*)malloc(size);
    
    
    for (int i = 0; i < WIDTH * WIDTH; ++i) {
        M[i] = rand() % 10;
        N[i] = rand() % 10;
    }
 

    gettimeofday(&start, NULL);

    MatrixMulOnHost(M, N, P, WIDTH);

    gettimeofday(&end, NULL);


    printf("Result Matrix:");
    for (int j =0 ; j<WIDTH*WIDTH ; j=j+1){
        if (j%WIDTH == 0){printf("\n %d \t",*(P+j));}
        else{printf("%d\t",*(P+j));}
    }

    printf("\n");
    printf("time spent: %0.8f sec\n", time_diff(&start, &end));

    free(M);
    free(N);
    free(P);
    return 0;
}