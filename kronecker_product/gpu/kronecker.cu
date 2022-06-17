// calculates the kronecker product of matrix a (n*n) and matrix b (p*p) 

#include <math.h>
#include <vector>
#include <iostream>
#include <cassert>

using namespace std;

__global__ void kronecker(int *a, int *b, int *c, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // C = [a_0 * B, a_1*B, a_2*B, ... a_(n*m)*B] | B = [b_0, b_1, b_2, ... b_(p*q)] 
    int a_index = (col / p) + (n * (row / p));
    int b_index = (col % p)      + (p * (row % p));
    c[col + row*(n*p)] = a[a_index] * b[b_index];
}

void verify_kronecker(vector<int> &a, vector<int> &b, vector<int> &c, int n, int p) {
    // iterates through every element of c and calculates verify_kronecker product for each element
    for (int j = 0; j < n*p; j++) {
        for (int i = 0; i < n*p; i++) {
            // C = [a_0 * B, a_1*B, a_2*B, ... a_(n*m)*B] | B = [b_0, b_1, b_2, ... b_(p*q)] 
            int a_index = floor(i / p) + (n * floor(j / p));
            int b_index = (i % p)      + (p * (j % p));
            assert( c[i + j*(n*p)] == a[a_index] * b[b_index] );
        }
    }
}

void display(vector<int> &M, int n) {
    // Display matrix
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            cout << M[j*n + i] << "\t";
        cout<<endl;
    }
}


int main() {
    int n = 1 << 2; // dim a
    int p = 1 << 3; // dim b
    vector<int> a(n * n);
    vector<int> b(p * p);
    vector<int> c(n * p * n * p);

    // initialize a and b with random values
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            a[i + j*n] = rand()%10;
        }
    }
    for (int j = 0; j < p; j++) {
        for (int i = 0; i < p; i++) {
            b[i + j*p] = rand()%10;
        }
    }

    // GPU code:
    // #################

    size_t a_bytes = n * n * sizeof(int);
    size_t b_bytes = p * p * sizeof(int);
    size_t c_bytes = n * n * p * p * sizeof(int);

    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    cudaMemcpy(d_a, a.data(), a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), b_bytes, cudaMemcpyHostToDevice);

    int THREADS = 32; // multiple of 32
    if (n*p % THREADS != 0) {
        cout << "matrix size (" << n*p << ") not divisible by THREADS (" << THREADS << ")" << endl;
        abort();
    }
    int BLOCKS = n*p / THREADS;

    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);

    kronecker<<<blocks, threads>>>(d_a, d_b, d_c, n, p);
    cudaMemcpy(c.data(), d_c, c_bytes, cudaMemcpyDeviceToHost);

    // #################

    verify_kronecker(a, b, c, n, p);

    cout << "\n-------------------- Matrix A :\n" << endl;
    display(a, n);
    cout << "\n-------------------- Matrix B :\n" << endl;
    display(b, p);
    cout << "\n-------------------- Matrix C :\n" << endl;
    display(c, n*p);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}

