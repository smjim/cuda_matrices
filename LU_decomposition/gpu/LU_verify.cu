// Performs LU decomposition on matrix A into L and U matrices, then re-multiplies to verify accuracy 
// Original gpu matrix multiplier by: nick from CoffeeBeforeArch

#include <algorithm>
#include <iostream>
#include <vector>
#include "LU_cpu.hpp"

using namespace std;

__global__ void matrixMul(float *a, float *b, float *c, int n) {
	// Compute each thread's global row and column index
	int row = blockIdx.y * blockDim.y + threadIdx.y; 
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// Iterate over row, and down column
	c[row * n + col] = 0;
	for (int k = 0; k < n; k++) {
		// Accumulate results for a single element
		c[row * n + col] += a[row * n + k] * b[k * n + col];
	}
}

int main() {
	// Matrix size of n * n (must divide into THREADS evenly)
	int n = 1 << 6;

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


	// Size (in bytes) of matrix
	size_t bytes = n * n * sizeof(float);

	// Allocate device memory
	float *d_a, *d_b, *d_c;
	cudaMalloc(&d_a, bytes);
	cudaMalloc(&d_b, bytes);
	cudaMalloc(&d_c, bytes);

	cudaMemcpy(d_a, L.data(), bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, U.data(), bytes, cudaMemcpyHostToDevice);

	// Threads per CTA dimension
	int THREADS = 32;

	// Blocks per grid dimension (assumes THREADS divides n evenly)
	if (n % THREADS != 0) {
		cout << "matrix size (" << n << ") not divisible by THREADS (" << THREADS << ")" << endl;
		abort();
	}
	int BLOCKS = n / THREADS;

	// Use dim3 structs for block	and grid dimensions
	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	// Launch kernel
	matrixMul<<<blocks, threads>>>(d_a, d_b, d_c, n);
	cudaDeviceSynchronize();

	cudaMemcpy(A.data(), d_c, bytes, cudaMemcpyDeviceToHost);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Display matrix product (should be == to original A matrix)
	cout << "\n-------------------- Product matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++)
			cout << floor(A[j*n + i] * 100.0) / 100.0 << "\t";
		cout<<endl;
	}	 

	return 0;
}
