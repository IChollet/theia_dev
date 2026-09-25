# FMM operators tutorial

### Choosing the interpolation
FMM operators (P2M, M2M, M2L) are free functions of the `theia` namespace. The interpolation method is selected by the type of the last argument, an empty *info* structure:
```c
#include "***/theia_dev/theia.hpp"
const theia::op_interpolation<DIM,FLT,T,ITYPE>        info;  // dense P2M matrices
const theia::op_tensor_interpolation<DIM,FLT,T,ITYPE> tinfo; // P2M kept in tensor form
```
with DIM the dimension, FLT the floating point type of the particles, T the type of the kernel values (e.g. `double` or `std::complex<double>`), and ITYPE the interpolation nodes (0 for Chebyshev, 1 for equispaced).

A cluster is described by its bounding box (`FLT mins[DIM]`, `FLT maxs[DIM]`), its particles (`std::array<FLT,DIM>*`) and its interpolation orders, given per dimension:
```c
int L[DIM] = {6, 11, 15}; // L[k] interpolation nodes along dimension k
```
Orders may differ between dimensions and between clusters. The L[0]*...*L[DIM-1] interpolation nodes form a tensor grid, the last dimension running fastest. All matrices are stored in column-major order.

### P2M (and L2P)
```c
T* P2M = nullptr;                       // allocated by the call if nullptr (to be freed with delete[])
theia::P2M(mins, maxs, N, particles, L, P2M, info);
```
`P2M` is the (prod L) x N matrix of the Lagrange polynomials of the cluster evaluated at its N particles. It is used as P2M on sources and, transposed, as L2P on targets:
```c
theia::gemm (1., P2M_y, q, 0., m, Lyd, Ny, 1);   // multipole  m = P2M_y q
theia::gemTm(1., P2M_x, l, 0., p, Nx, Lxd, 1);   // potential  p = P2M_x^T l
```

### M2L
```c
T* M2L = nullptr;                       // allocated if nullptr, (prod Lx) x (prod Ly)
theia::M2L(minsX, maxsX, Lx, minsY, maxsY, Ly, K, M2L, info);
```
`K` is a user functor evaluating the kernel between two sets of points:
```c
void operator()(std::array<FLT,DIM>* X, int Nx, std::array<FLT,DIM>* Y, int Ny, T* A); // A[j*Nx+i] = K(X[i],Y[j])
```

### M2M (and L2L)
```c
theia::Kron<DIM,T> M2M;
theia::M2M(L_parent, mins_parent, maxs_parent, L_child, mins_child, maxs_child, M2M, info);
gemm (M2M, m_child, m_parent, nrhs);    // M2M : multipole of the child -> parent
gemTm(M2M, l_parent, l_child, nrhs);    // L2L : local expansion of the parent -> child
```
M2M is stored as a Kronecker product of DIM small matrices (parent polynomials at the child nodes), of size L_parent[k] x L_child[k]. `Kron` keeps its own copy of the matrices and sizes.

### Tensor form of P2M
With `op_tensor_interpolation`, P2M only stores the DIM matrices of 1D polynomials (L[k] x N each) and computes the multivariate polynomials particle by particle (no dense (prod L) x N matrix):
```c
theia::TensorP2M<DIM,T> P2M_y;
theia::P2M(mins, maxs, N, particles, L, P2M_y, tinfo);
gemm (P2M_y, q, m, nrhs);               // P2M
gemTm(P2M_x, l, p, nrhs);               // L2P
P2M_y.to_dense(S);                      // same matrix as the dense P2M (S allocated if nullptr)
```
M2L and M2M are the same as with `op_interpolation`.

The product of a `Kron` by a `TensorP2M` is again a `TensorP2M` (mixed-product property, cost DIM small matrix products):
```c
theia::TensorP2M<DIM,T> T_y;
gemm (M2M, P2M_y, T_y);                 // T_y = M2M * P2M_y    : P2M directly onto the parent
gemTm(K,   P2M_y, T_y);                 // T_y = K^T * P2M_y
gemTm(T_y, l_parent, p, nrhs);          // L2P o L2L = (M2M * P2M)^T
```
When the parent orders are lower or equal to the child ones, M2M * P2M_child is exactly the P2M of the parent.

### Performance notes
- The dense P2M costs (prod L) * N in memory; once computed, BLAS products are the fastest way to apply it.
- The tensor form costs sum(L) * N in memory and is applied matrix-free (same number of flops): prefer it when memory is the limiting factor, or to build compositions of operators (M2M * P2M, ...).

### Examples
See "test/interpolation/test_interpolation_double.cpp", "test_reinterpolation_double.cpp", "test_tensor_*", and "test_anisotropic_*".
