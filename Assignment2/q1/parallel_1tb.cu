%%cuda
// Author: Arnav Gautam
// ID: 12140280

// Q1, Version - II

#include <bits/stdc++.h>
using namespace std;
using namespace chrono;

#define BLOCK_SIZE 4

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

int getRandomNumber(int l, int r) {
	return uniform_int_distribution<int>(l, r)(rng);
}

__global__ void dkernel(int *M, int *N, int m) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int dx[4] = {1, 0, -1, 0}; // for 4 directional transition as stated in the problem
	const int dy[4] = {0, 1, 0, -1}; // same as above
	if (i < m && j < m) {
		for (int ptr = 0; ptr < 4; ptr++) {
			int X = i + dx[ptr];
			int Y = j + dy[ptr];
			N[i * m + j] += M[X * m + Y];
		}
	}
}

int main() {
	int m = 4; // assume
	int M[m][m]; // given matrix
	int ans[m][m]; // output matrix
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			M[i][j] = getRandomNumber(1, 10); // assume elements are from 1 to 10
		}
	}
	cout << "Original Matrix" << '\n';
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			cout << M[i][j] << " ";
		}
		cout << '\n';
	}
	cout << "-----------------------------------\n";
	int *d_M, *d_ans;
	cudaMalloc(&d_M, m * m * sizeof(int));
	cudaMalloc(&d_ans, m * m * sizeof(int));
	cudaMemcpy(d_M, M, m * m * sizeof(int), cudaMemcpyHostToDevice);

	dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridSize((m + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);

	dkernel<<<gridSize, blockSize>>>(d_M, d_ans, m); // compute it

	cudaMemcpy(ans, d_ans, m * m * sizeof(int), cudaMemcpyDeviceToHost);

	cudaFree(d_M); cudaFree(d_ans);

	cout << "Answer Matrix" << '\n';

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			cout << ans[i][j] << " ";
		}
		cout << '\n';
	}

	return 0;
}