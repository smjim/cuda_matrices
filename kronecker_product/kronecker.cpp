// calculates the kronecker product of matrix a (n*m) and matrix b (p*q)	

#include <math.h>
#include <vector>
#include <iostream>

using namespace std;

void kronecker(vector<int> &a, vector<int> &b, vector<int> &c, int n, int m, int p, int q) {
	// iterates through every element of c and calculates kronecker product for each element
	for (int j = 0; j < n*p; j++) {
		for (int i = 0; i < m*q; i++) {
			// C = [a_0 * B, a_1*B, a_2*B, ... a_(n*m)*B] | B = [b_0, b_1, b_2, ... b_(p*q)] 
			int a_index = floor(i / q) + (m * floor(j / p));
			int b_index = (i % q)      + (q * (j % p));
			c[i + j*(m*q)] = a[a_index] * b[b_index]; 
		}
	}	
}

void display(vector<int> &A, int n, int m) {
    // Display matrix
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++)
            cout << A[j*m + i] << "\t";
        cout<<endl;
    }   
}


int main() {
	int n = 3;	// height a
	int m = 3;	// width a
	int p = 5;	// height b
	int q = 2;	// width b
	vector<int> a(n * m);
	vector<int> b(p * q);
	vector<int> c(n * p * m * q);

	// initialize a and b with random values
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < m; i++) {
			a[i + j*m] = floor(rand()%10);
		}
	}
	for (int j = 0; j < p; j++) {
		for (int i = 0; i < q; i++) {
			b[i + j*q] = floor(rand()%10);
		}
	}

	// a (kronecker) b = c
	kronecker(a, b, c, n, m, p, q);

    cout << "\n-------------------- Matrix A :\n" << endl;
	display(a, n, m);
    cout << "\n-------------------- Matrix B :\n" << endl;
	display(b, p, q);
    cout << "\n-------------------- Matrix C :\n" << endl;
	display(c, n*p, m*q);

	return 0;
}
