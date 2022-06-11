// This code multiplies two given matrices 
// Original gpu matrix multiplier by: nick from CoffeeBeforeArch

#ifndef MULTIPLY_INCLUDED
#define MULTIPLY_INCLUDED

#include <math.h>
#include <cassert>

using namespace std;

__global__ void matrixMul(float *a, float *b, float *c, int n) {
    // Compute each thread's global row and column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    tmp = 0;
    for (int k = 0; k < n; k++) {
        // Accumulate results for a single element
        tmp += a[row * n + k] * b[k * n + col];
    }   
	
	assert(c[row * n + col] == tmp);
}


#endif // MULTIPLY_INCLUDED
