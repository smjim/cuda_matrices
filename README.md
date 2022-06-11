# cuda_matrices
implementations of common matrix operations in cuda c++



# Future projects:

* using LU decomposition program to make a cuda-accelerated matrix inverter
* cuda-accelerated determinant calculator
* cuda-accelerated image processing/ matrix kernel operations
* eigenvalue/ eigenvector calculator (if it can be parallelized)

![imagen](https://user-images.githubusercontent.com/78174712/172536011-3bba3d63-e902-40c4-8910-df7a4556708f.png)
(from \[1])

![imagen](https://user-images.githubusercontent.com/78174712/172536306-ecb1cee8-8278-43de-accc-1726e28f17b3.png)
(LDU decomposition of a Walsh matrix)

# Interesting notes:

* The so-called "Hilbert Matrix" is infamous for its non-singularity for any size n, and is an excellent candidate for testing the numerical stability of any matrix algorithm

### Papers I've investigated:

1. Sadal Anisul Haque, Marc Moreno Maza. _Determinant Computation on the GPU using the Condensation Method_. https://sci-hub.st/10.1088/1742-6596/341/1/012031
2. Jack Dongarra et al. High-performance Cholesky factorization for GPU-only execution. https://www.icl.utk.edu/files/publications/2017/icl-utk-987-2017.pdf
