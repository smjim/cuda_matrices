#include <vector>

// function sets matrix A = to A * transpose(A)
void transpose_multiply(vector<double> &a, int n) {
	vector<double> tmp(n * n);
    for (int k = 0; k < n; k ++) {
        for (int i = 0; i < n; i++) {
            double sum = 0;    
            // for every row-column pair  
            for (int j = 0; j < n; j++) {
                // accumulate sum of row-column pair products
                sum += a[k*n + j] * a[i*n + j]; 
            }
            tmp[i*n + k] = sum;
			tmp[k*n + i] = sum;
        }
    }  

	for (int i = 0; i < n*n; i++) {
		a[i] = tmp[i];
	}

}
