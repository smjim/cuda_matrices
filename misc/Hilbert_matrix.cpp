// Generates a Hilbert matrix

#include <iostream>
#include <vector>
#include <math.h>

using namespace std;

void hilbert(vector<float> &h, int n) {
	for (int j = 0; j < n; j++) {
		for (int i = 0; i < n; i++) {
			h[i + j*n] = 1./(i + j + 1.);
			//cout << 1./ (i + j + 1.) << endl;
		}
	} 
}

int main() {
	// Matrix size of n * n
	int n = 1 << 3;

	// Initialize matrix H to store data 
	vector<float> H(n*n);

	// Create the Hilbert matrix
	hilbert(H, n);	

	// Display matrix 
	cout << "\n-------------------- H matrix:\n" << endl;
	for (int j = 0; j < n; j++) {
			for (int i = 0; i < n; i++)
					cout << floor(H[j*n + i] * 100.0) / 100.0 << "\t";
			cout<<endl;
	}	 

	return 0;
}
