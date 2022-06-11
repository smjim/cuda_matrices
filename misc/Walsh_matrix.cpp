// Generates a Walsh matrix

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

void kronecker(vector<int> &a, vector<int> &b, vector<int> &c, int n) { 
    // iterates through every element of c and calculates kronecker product for each element
    for (int j = 0; j < 2*n; j++) {
        for (int i = 0; i < 2*n; i++) {
            // C = [a_0 * B, a_1*B, a_2*B, ... a_(n*m)*B] | B = [b_0, b_1, b_2, ... b_(p*q)] 
            int a_index = floor(i / n) + (2 * floor(j / n));
            int b_index = (i % n)      + (n * (j % n));
            c[i + j*(2*n)] = a[a_index] * b[b_index]; 
        }
    }   
}

int main() {
	// # Iterations (Walsh(N) is size 2^(N+1) * 2^(N+1))
	int N = 3;
	int dim = pow(2, N+1); // # elements/ row in Walsh(N)

	// Initialize matrix W to store data 
	vector<int> w_0(4);
	vector<int> W(dim*dim);

	// Walsh(0) is defined as:
	w_0 = {1, 1, 1, -1};
	W = w_0;

	// Perform the Kronecker operation N times
	for (int i = 0; i < N; i++) {
		int n = pow(2, i+1);
		vector<int> tmp(pow(n, 4));

		kronecker(w_0, W, tmp, n);
		W = tmp;
	}	

	// Display matrix 
	cout << "\n-------------------- W matrix:\n" << endl;
	for (int j = 0; j < dim; j++) {
		for (int i = 0; i < dim; i++)
			cout << W[j*dim + i] << "\t";
		cout<<endl;
	}	 

	return 0;
}
