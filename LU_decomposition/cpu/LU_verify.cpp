// Performs LU decomposition on matrix A into L and U matrices, then re-multiplies to verify accuracy 
// Original cpu matrix multiplier by: nick from CoffeeBeforeArch

#include <algorithm>
#include <iostream>
#include <vector>
#include "LU_cpu.hpp"

using namespace std;

void verify_result(vector<float> &a, vector<float> &b, vector<float> &c, int n) {
    // For every row...
    for (int j = 0; j < n; j++) {

        // For every column...
        for (int i = 0; i < n; i++) {

            float tmp = 0;
            // For every element in the row-column pair
            for (int k = 0; k < n; k++) {

                // Accumulate the partial results
                tmp += a[j * n + k] * b[k * n + i]; 
            }
            c[j * n + i] = tmp;
        }    
    }   
}

int main() {
	// Matrix size of n * n
	int n = 1 << 10;

	// Declare matrix vectors
	vector<float> A(n*n);
	vector<float> L(n*n);
	vector<float> U(n*n);

	// Initialize matrix A with random values
	for (int j = 0; j < n; j++) 
			for (int i = 0; i < n; i++)
					A[j*n + i] = rand()%100 + 1;

	// Display matrix A
	cout << "\n-------------------- A matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
			for (int i = 0; i < n; i++)
					cout << floor(A[j*n + i] * 100.0) / 100.0 << "\t";
			cout<<endl;
	}	 

	// Decompose A into L and U matrices
	factorize(A, L, U, n); 

	// Display L and U matrices
	cout << "\n-------------------- L matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
			for (int i = 0; i < n; i++)
					cout << floor(L[j*n + i] * 100.0) / 100.0 << "\t";
			cout<<endl;
	}	 

	cout << "\n-------------------- U matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
			for (int i = 0; i < n; i++)
					cout << floor(U[j*n + i] * 100.0) / 100.0 << "\t";
			cout<<endl;
	}	

	cout << "\n##########################################" << endl;

	verify_result(L, U, A, n);

	// Display matrix product (should be == to original A matrix)
	cout << "\n-------------------- Product matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++)
			cout << floor(A[j*n + i] * 100.0) / 100.0 << "\t";
		cout<<endl;
	}	 

	return 0;
}
