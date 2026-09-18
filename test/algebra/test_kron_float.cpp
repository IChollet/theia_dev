#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/theia.hpp"

#define FLT float
#define DIM 3
#define urand rand()/FLT(RAND_MAX)

int main(){

  // Dimensions of matrices
  int M[DIM] = {7, 27, 7};
  int N[DIM] = {6, 24, 18};
  int MM = M[0]*M[1]*M[2];
  int NN = N[0]*N[1]*N[2];

  // Initialise (randomly) the matrices
  FLT** A = new FLT*[DIM];
  FLT * q = new FLT[NN];
  FLT * r = new FLT[MM];
  FLT * e = new FLT[MM];
  for(int i = 0; i < DIM; i++){
    A[i] = new FLT[M[i]*N[i]];
    for(int ii = 0; ii < M[i]; ii++){
      for(int jj = 0; jj < N[i]; jj++){
	A[i][ii + jj*M[i]] = urand;
      }
    }
  }
  for(int i = 0; i < NN; i++){q[i] = FLT(urand);}
  for(int i = 0; i < MM; i++){r[i] = FLT(urand);}

  // Get Kronecker struct
  theia::Kron<DIM,FLT> K(A,M,N);
  
  // Kronecker struct product
  gemm(K,q,r,1);
  
  // Get naive product
  theia::nkmv(A, M, N, q, e, DIM);

  // Verify result
  double num = 0.;
  double div = 0.;
  for(int i = 0; i < MM; i++){
    num += std::abs(e[i]-r[i]);
    div += std::abs(e[i]);
  }

  std::cout << std::boolalpha << (std::abs(num/div) < 1.e-6) << std::endl;
  
  return 0;
}
