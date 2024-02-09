
// author: Arnav Gautam

#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

int main() {


	// #ifndef ONLINE_JUDGE // Please ignore these comments, its private to my system
	// freopen("input.txt", "r", stdin);
	// freopen("output.txt", "w", stdout);
	// #endif

	int N, R;
	// cin >> N >> R; // take N as number of elements in array and R as the radius as given in the problem
	N = 131072;
	R = 32;
	auto start1 = high_resolution_clock::now();
	int* A = new int[N]; // using to store memory in heap to prevent stack overflow due to large N
	// for (int i = 0; i < N; i++) cin >> A[i]; // taking array A
	for (int i = 0; i < N; i++) A[i] = i;

	// we need to compute A[i - r] + A[i + r] from r = 0..r = R

	int* B = new int[N];
	for (int i = 0; i < N; i++) {
		int sm = 0;
		for (int r = 0; r < R; r++) {
			if (r == 0) {
				sm += A[i];
				continue;
			}
			if (i - r >= 0) {
				sm += A[i - r];
		}
			if (i + r <= N - 1) {
				sm += A[i + r];
			}
		}
		B[i] = sm;
	}

	// for (int i = 0; i < N; i++) {
	// 	cout << B[i] << " ";
	// }
	// cout << '\n'; // for printing the elements of array B

	auto stop1 = high_resolution_clock::now();

	auto duration = duration_cast<microseconds>(stop1 - start1);
	cout << "Time: " << duration.count()/1e6 << endl; // this is seconds

	return 0;
}