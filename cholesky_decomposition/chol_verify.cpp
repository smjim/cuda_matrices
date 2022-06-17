// Performs cholesky decomposition on matrix A into L and L' matrices, then re-multiplies to verify accuracy 

#include <iostream>
#include <vector>
#include "chol_cpu.hpp"
#include "transpose_cpu.hpp"

using namespace std;

// asserts that l times l' == a, assumes l is lower triangular
void verify_result(vector<double> &l, vector<double> &a, int n) {

    for (int j = 0; j < n; j ++) {
        for (int k = 0; k < n; k++) {
            double tmp = 0;     
            // for every row-column pair (excluding l[i, j] == 0)   
            for (int i = 0; i < n; i++) {
                // accumulate sum of row-column pair products
                tmp += l[j*n + i] * l[k*n + i]; 
            }
            a[k*n + j] = tmp;
        }   
    }
}

int main() {
	// Matrix size of n * n
	int n = 1 << 4;

	// Declare matrix vectors
	vector<double> A(n*n);
	vector<double> L(n*n);

	// Initialize matrix A to be symmetric positive definite 
	for (int j = 0; j < n; j++) 
		for (int i = 0; i < n; i++)
			A[j*n + i] = rand()%20 - 10;
	// A = A * transpose(A) will guarantee symmetric positive definite
	transpose_multiply(A, n);

	// Display matrix A
	cout << "\n-------------------- A matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++)
			cout << floor(A[j*n + i] * 100.0) / 100.0 << "\t";
		cout<<endl;
	}	 

	// Decompose A into L and L' matrices
	cholesky(A, L, n); 

	// Display L and L' matrices
	cout << "\n-------------------- L matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++)
			cout << floor(L[j*n + i] * 100.0) / 100.0 << "\t";
		cout<<endl;
	}	 

	cout << "\n-------------------- L' matrix:\n" << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++)
			cout << floor(L[j*n + i] * 100.0) / 100.0 << "\t";
		cout<<endl;
	}	

	cout << "\n##########################################" << endl;

	verify_result(L, A, n);

	// Display matrix product (should be == to original A matrix)
	cout << "\n-------------------- Product matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++)
			cout << floor(A[j*n + i] * 100.0) / 100.0 << "\t";
		cout<<endl;
	}	 

	return 0;
}
