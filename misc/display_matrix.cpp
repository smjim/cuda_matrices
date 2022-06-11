// Displays a matrix

#ifndef DISPLAY_INCLUDED
#define DISPLAY_INCLUDED

#include <iostream>

using namespace std:

void display(vector<float> &A, int n) {
    // Display matrix A
    cout << "\n-------------------- Matrix A :\n" << endl;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++)
            cout << floor(A[j*n + i] * 100.0) / 100.0 << "\t";
        cout<<endl;
    } 
}

#endif // DISPLAY_INCLUDED
