// Finds the determinant of a matrix of size n * n using "salem and said" condensation method outlined in paper

#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <cassert>

using namespace std;

__global__ int det(int *a, int pivot, int l, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	// initialize b
	vector<int> b((n-1)*(n-1));

	// define b
	if (row < l) {
		b[col + row*(n-1)] = -a[(col+1) + row*n] * a[l*n];
	} else { // row >= l
		b[col + row*(n-1)] = (a[l*n] * a[(col+1) + (row+1)*n]) - (a[(row+1)*n] * a[(col+1) + l*n]);
	}

	// define pivot of b to call recursively
	l = find_pivot<<<1, 1>>>(b, pivot, n);

	// find determinant of b
	int b_det;
	if (n-1 == 2) { // if n is small then find det(b) through formula
		b_det = b[0]*b[3] - b[1]*b[2];
	} else {		// find det(b) through recursion

		// Threads per CTA dimension
		int THREADS = 4;
    	int BLOCKS = n / THREADS;

		dim3 blocks = (BLOCKS, BLOCKS);
		dim3 threads = (THREADS, THREADS);
		b_det = det<<<blocks, threads>>>(b, pivot, l, n-1);
	}

	// use determinant of b to find determinant of a
	int a_det = b_det / pow(a[l*n], n-2);

	// return determinant of a
	return a_det;	
}

__global__ int find_pivot(int *a, int pivot, int n) {
	for (int i = 0; i < n; i++) {
		if (a[i] == 0) {
			pivot *= a[i];
			return i;
		}
	}	
	return -1; // couldnt find a pivot in first row, det(A) = 0
}

// from condensation method paper, also try montgomery reduction to see if it works better
// p = small (<30 bit) prime, pinv = 1/p
__device__ int double_mul_mod(int a, int b, int p, double pinv) {
	int q = (int) ((((double) a) * ((double) b)) * pinv);
	int res = a * b - q * p;
	return (res < 0) ? (-res) : res;
}

void verify(vector<float> &a, int n, int det) {
	float tmp = 0;

	if (n == 2) {
		tmp = (a[0]*a[3]) - (a[1]*a[2]);
	} else {
		for (int k = 0; k < n; k++) {
			vector<float> b((n-1) * (n-1));

			for (int j = 0; j < n; j++) {
				for (int i = 0; i < n; i++) {
					// cofactor expansion along a[k][0]
					if (i > k && j > 0) 
						b[(j-1)*(n-1) + i-1] = a[j*n + i];
					else if (i < k && j > 0)
						b[(j-1)*(n-1) + i] = a[j*n + i];
				}
			}

			tmp += a[k] * pow(-1, k) * det(b, n-1);
		}
	}

	assert(tmp == det);
}

int main() {
	// Matrix size of n * n
	// using cofactor expansion, problem is O(n!) complex
	// using optimized condensation method, problem is O(n^3) complex
	int n = 4;//1<<3; 

	// Declare matrix vector
    vector<int> A(n*n);

    // Initialize matrix A with random values
    for (int j = 0; j < n; j++) 
        for (int i = 0; i < n; i++)
            A[j*n + i] = rand()%100 + 1;

    // Display matrix A
    cout << "\n-------------------- Matrix A :\n" << endl;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            cout << floor(A[j*n + i] * 100.0) / 100.0 << "\t";
        cout<<endl;
    }   

	// Cuda code below
	// #########################

	// Allocate device memory and move data
	size_t bytes = n * n * sizeof(int);

	float *d_a;
	cudaMalloc(&d_a, bytes);
	cudaMemcpy(d_a, A.data(), bytes, cudaMemcpyHostToDevice);

	// Threads per CTA dimension
	int THREADS = 4;

	// Blocks per grid dimension (assumes THREADS divides n evenly)
    if (n % THREADS != 0) {
        cout << "matrix size (" << n << ") not divisible by THREADS (" << THREADS << ")" << endl;
        abort();
    }   
    int BLOCKS = n / THREADS;

	dim3 threads(THREADS, THREADS);
	dim3 blocks(BLOCKS, BLOCKS);

	// Find pivot in first row
	int pivot = 1;	// product of pivots 
	int l; 			// index of pivot
	l = find_pivot<<<1, 1>>>(d_a, pivot, n);
	cudaDeviceSynchronize();
	if (l == -1) {
		cudaFree(d_a);
		cout << "first row of matrix is NULL vector, det = 0" << endl;
		abort();
	}

    // Compute determinant of A 
	//det<<<blocks, threads>>>(A, n);
	int determinant = det<<<blocks, threads>>>(d_a, pivot, l, n);
    cout << "\ndet(A) = " << determinant << endl; 

	verify(A, n, determinant);

	cudaFree(d_a);

	return 0;
}
