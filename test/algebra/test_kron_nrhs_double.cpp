#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/theia.hpp"  

#define FLT double
#define DIM 3
#define urand rand()/FLT(RAND_MAX)

// Kron products with several right-hand sides, compared to the naive product column by column
int main(){
  srand(235);
  int M[DIM] = {7, 27, 7};
  int N[DIM] = {6, 24, 18};
  int MM = M[0]*M[1]*M[2];
  int NN = N[0]*N[1]*N[2];
  int nrhs = 3;
  FLT** A = new FLT*[DIM];
  for(int i = 0; i < DIM; i++){
    A[i] = new FLT[M[i]*N[i]];
    for(int j = 0; j < M[i]*N[i]; j++){A[i][j] = urand;}
  }
  FLT *q  = new FLT[NN*nrhs], *r  = new FLT[MM*nrhs], *e  = new FLT[MM];
  FLT *qT = new FLT[MM*nrhs], *rT = new FLT[NN*nrhs], *eT = new FLT[NN];
  for(int i = 0; i < NN*nrhs; i++){q [i] = urand;}
  for(int i = 0; i < MM*nrhs; i++){qT[i] = urand;}

  // Kron keeps its own copy: input arrays can be modified afterwards
  theia::Kron<DIM,FLT> K(A,M,N);
  gemm (K,q ,r ,nrhs);
  gemTm(K,qT,rT,nrhs);

  double num = 0., div = 0.;
  for(int k = 0; k < nrhs; k++){
    theia::nkmv (A, M, N, q +k*NN, e , DIM);
    theia::nkmTv(A, M, N, qT+k*MM, eT, DIM);
    for(int i = 0; i < MM; i++){num += std::abs(e [i]-r [k*MM+i]); div += std::abs(e [i]);}
    for(int i = 0; i < NN; i++){num += std::abs(eT[i]-rT[k*NN+i]); div += std::abs(eT[i]);}
  }
  std::cout << std::boolalpha << (num/div < 1.e-14) << std::endl;

  for(int i = 0; i < DIM; i++){delete [] A[i];}
  delete [] A; delete [] q; delete [] r; delete [] e; delete [] qT; delete [] rT; delete [] eT;
  return 0;
}
