
// Author: Arnav Gautam
// ID: 12140280

// Q1, Version - I

#include <bits/stdc++.h>

using namespace std;
using namespace chrono;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const int dx[4] = {1, 0, -1, 0}; // for 4 directional transition as stated in the problem
const int dy[4] = {0, 1, 0, -1}; // same as above

int getRandomNumber(int l, int r) {return uniform_int_distribution<int>(l, r)(rng);} 

int main() {

	// PLEASE IGNORE THESE BELOW 4 COMMENTS, THESE ARE FOR I/O IN MY LOCAL COMPILER.
	#ifndef ONLINE_JUDGE
	freopen("input.txt", "r", stdin);
	freopen("output.txt", "w", stdout);
	#endif

	int m;
	cin >> m;
	
	int M[m][m]; // given matrix

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			M[i][j] = getRandomNumber(1, 10); // assigning random number from 1 to 10
		}
	}

	int N[m][m]; // matrix to compute
	for (int i = 0; i < m; i++) { 
		for (int j = 0; j < m; j++) {
			N[i][j] = 0; // initializing the output matrix N
		} 
	}

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			for (int ptr = 0; ptr < 4; ptr++) {
				int X = i + dx[ptr];
				int Y = j + dy[ptr];
				if (X >= m || X < 0 || Y >= m || Y < 0) continue;
				N[i][j] += M[X][Y]; // transition to be performed as given in the problem
			}
		}
	}

	cout << "Original Matrix:\n";

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			cout << M[i][j] << " ";
		}
		cout << '\n';
	}

	cout << "-------------------------------------\n"; // for better vision of output

	cout << "Computed Matrix:\n";

	for (int i = 0; i < m; i++) {
		for (int j = 0; j < m; j++) {
			cout << N[i][j] << " ";
		}
		cout << '\n';
	}

	return 0;
}