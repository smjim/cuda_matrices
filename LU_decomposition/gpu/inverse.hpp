// This code finds the LU factorization of an nxn matrix using cpu code

#ifndef FACTORIZE_INCLUDED
#define FACTORIZE_INCLUDED

#include <math.h>

using namespace std;

void factorize(vector<float> &a, vector<float> &l, vector<float> &u, int n)
{
    // function transforms matrix A into matrix U, 
    // taking row reduction coefficients into matrix L
    // u[j*n + i]' = a[j*n + i] - (a[j*n+k] / a[k][k]) * a[k*n+i]

    // for every pivot
    for (int k = 0; k < n; k++) {

        // for every column
        for (int j = 0; j < n; j++) {
            if (j < k)
                l[j*n + k] = 0;
            else {
                l[j*n + k] = a[j*n + k]; 
                // for every row
                for (int i = 0; i < k; i++) {
                    l[j*n + k] -= l[j*n + i] * u[i*n + k]; 
                }
            }
        }

        for (int j = 0; j < n; j++) {
            if (j < k)
                u[k*n + j] = 0;
            else if (j == k)
                u[k*n + j] = 1;
            else {
                u[k*n + j] = a[k*n + j] / l[k*n + k]; 
                for (int i = 0; i < k; i++) {
                    u[k*n + j] -= l[k*n + i] * u[i*n + j] / l[k*n + k]; 
                }
            }
        }
    }  

}

#endif // FACTORIZE_INCLUDED
