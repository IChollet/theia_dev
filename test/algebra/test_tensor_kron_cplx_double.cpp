#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/theia.hpp"  

#define FLT double
#define VAL std::complex<FLT>
#define DIM 3
#define urand rand()/FLT(RAND_MAX)

// TensorP2M algebra with anisotropic sizes, compared to dense / Kron computations:
//   gemm(A,B), gemTm(A,B)            (A tensor, B dense, several right-hand sides)
//   gemm(K,A) = K A, gemTm(K,A) = K^T A   (K Kron, A tensor)
VAL rnd(){return VAL(urand-0.5);}

int main(){
  srand(235);
  int Ls[DIM] = {4, 6, 5};    // orders of A
  int Ms[DIM] = {5, 7, 3};    // rows of K     (K_d : Ms[d] x Ls[d])
  int Ps[DIM] = {3, 8, 6};    // cols of K2    (K2_d: Ls[d] x Ps[d])
  int N = 37, nrhs = 3;
  int Ld = Ls[0]*Ls[1]*Ls[2], Md = Ms[0]*Ms[1]*Ms[2], Pd = Ps[0]*Ps[1]*Ps[2];

  VAL *F[DIM], *K[DIM], *K2[DIM];
  for(int d = 0; d < DIM; d++){
    F [d] = new VAL[Ls[d]*N];     for(int i = 0; i < Ls[d]*N;     i++){F [d][i] = rnd();}
    K [d] = new VAL[Ms[d]*Ls[d]]; for(int i = 0; i < Ms[d]*Ls[d]; i++){K [d][i] = rnd();}
    K2[d] = new VAL[Ls[d]*Ps[d]]; for(int i = 0; i < Ls[d]*Ps[d]; i++){K2[d][i] = rnd();}
  }
  theia::TensorP2M<DIM,VAL> A(F,Ls,N);
  theia::Kron<DIM,VAL> KK(K,Ms,Ls), KK2(K2,Ls,Ps);
  VAL* Ad = nullptr;
  A.to_dense(Ad);
  double err = 0., nrm = 0.;
  auto cmp = [&](VAL* x, VAL* y, int n){for(int i = 0; i < n; i++){err = std::max(err,std::abs(x[i]-y[i])); nrm = std::max(nrm,std::abs(y[i]));}};

  // A B and A^T B
  VAL *B  = new VAL[N *nrhs], *C  = new VAL[Ld*nrhs], *E  = new VAL[Ld*nrhs];
  VAL *BT = new VAL[Ld*nrhs], *CT = new VAL[N *nrhs], *ET = new VAL[N *nrhs];
  for(int i = 0; i < N *nrhs; i++){B [i] = rnd();}
  for(int i = 0; i < Ld*nrhs; i++){BT[i] = rnd();}
  gemm (A, B , C , nrhs); theia::gemm (1., Ad, B , 0., E , Ld, N , nrhs); cmp(C , E , Ld*nrhs);
  gemTm(A, BT, CT, nrhs); theia::gemTm(1., Ad, BT, 0., ET, N , Ld, nrhs); cmp(CT, ET, N *nrhs);

  // K A : column j of the result = Kron product applied to column j of A
  theia::TensorP2M<DIM,VAL> KA;
  gemm(KK, A, KA);
  VAL *R = nullptr, *Rd = new VAL[Md*N];
  KA.to_dense(R);
  gemm(KK, Ad, Rd, N);
  cmp(R, Rd, Md*N);

  // K2^T A
  theia::TensorP2M<DIM,VAL> K2TA;
  gemTm(KK2, A, K2TA);
  VAL *RT = nullptr, *RTd = new VAL[Pd*N];
  K2TA.to_dense(RT);
  gemTm(KK2, Ad, RTd, N);
  cmp(RT, RTd, Pd*N);

  // In place: A <- K A
  gemm(KK, A, A);
  VAL *R2 = nullptr;
  A.to_dense(R2);
  cmp(R2, Rd, Md*N);

  std::cout << std::boolalpha << (err/nrm < 1.e-13) << std::endl;

  for(int d = 0; d < DIM; d++){delete [] F[d]; delete [] K[d]; delete [] K2[d];}
  delete [] Ad; delete [] B; delete [] C; delete [] E; delete [] BT; delete [] CT; delete [] ET;
  delete [] R; delete [] Rd; delete [] RT; delete [] RTd; delete [] R2;
  return 0;
}
