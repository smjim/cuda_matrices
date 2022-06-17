// This code finds the cholesky factorization of an n*n matrix using cpu code
// matrix must be symmetric positive definite for cholesky decomposition to apply

#ifndef CHOLESKY_INCLUDED
#define CHOLESKY_INCLUDED

#include <cmath>

using namespace std;

void cholesky(vector<double> &a, vector<double> &l, int n) {
	l[0] = sqrt(a[0]);				// first element

	for (int j = 1; j < n; j++)
		l[j*n] = a[j*n] / l[0]; 	// first row

	for (int i = 1; i < n; i++) {
		double sum = 0;
		for (int k = 0; k < i; k++)
			sum += pow(l[k + i*n], 2);
		l[i*n + i] = sqrt(a[i*n + i] - sum);				// diagonals

		for (int j = i+1; j < n; j++) {
			sum = 0;
			for (int k = 0; k < i; k++)
				sum += l[k + i*n] * l[k + j*n];
			l[i + j*n] = (a[i + j*n] - sum) / l[i + i*n];	// everything else
		}
	}

}

#endif // CHOLESKY_INCLUDED
