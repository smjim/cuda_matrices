// Finds the determinant of a matrix of size n * n using cofactor expansion

#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>

using namespace std;

float det(vector<float> &a, int n) {
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

	return tmp;
}

int main() {
	// Matrix size of n * n
	int n = 8; // using cofactor expansion, problem is O(n!) complex

	// Declare matrix vector
    vector<float> A(n*n);

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

    // Compute determinant of A 
    cout << "\ndet(A) = " << det(A, n) << endl; 

	return 0;
}
