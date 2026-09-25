#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/theia.hpp"  

#define FLT double
#define VAL FLT
#define DIM 3
#define urand rand()/FLT(RAND_MAX)

class light{
public :
  void operator()(std::array<FLT,DIM>* X, int Nx, std::array<FLT,DIM>* Y, int Ny, VAL* A){
    for(int j = 0; j < Ny; j++){
      for(int i = 0; i < Nx; i++){
        FLT R0 = X[i][0] - Y[j][0];
	FLT R1 = X[i][1] - Y[j][1];
	FLT R2 = X[i][2] - Y[j][2];
	FLT R  = R0*R0 + R1*R1 + R2*R2;
	A[j*Nx+i] = VAL(.5 * exp(-.5*sqrt(R)) / R);
      }
    }
  }
};

/*
  Tensor P2M composed with M2M (and L2P composed with L2L = M2M^T):
    M2M * P2M_child                is a TensorP2M  (gemm(Kron,TensorP2M))
    L2P_child * L2L = (M2M * P2M_child)^T  is applied with gemTm on the same object
  Checks: far field vs direct evaluation, vs the step-by-step path,
  and nestedness (M2M * P2M_child == P2M_parent for equal orders).
*/
int main(){
  srand(235);
  int Nx = 374, Ny = 93, L = 12;
  std::array<FLT,DIM>* X = new std::array<FLT,DIM>[Nx];
  std::array<FLT,DIM>* Y = new std::array<FLT,DIM>[Ny];
  VAL *q = new VAL[Ny], *a = new VAL[Nx], *b = new VAL[Nx], *e = new VAL[Nx];
  light Kernel;

  // Leaf clusters and sampling
  FLT minsX[3] = {0.,0.,0.},  maxsX[3] = {1.,1.,1.};
  FLT minsY[3] = {0.2,0.1,6.}, maxsY[3] = {0.9,1.7,7.};
  for(int i = 0; i < Nx; i++){for(int k = 0; k < DIM; k++){X[i][k] = minsX[k] + urand*(maxsX[k]-minsX[k]);}}
  for(int i = 0; i < Ny; i++){for(int k = 0; k < DIM; k++){Y[i][k] = minsY[k] + urand*(maxsY[k]-minsY[k]);} q[i] = VAL(urand);}

  // Non-leaf clusters
  FLT minsXu[3] = {0.,0.,0.}, maxsXu[3] = {2.,2.,2.};
  FLT minsYu[3] = {0.,0.,6.}, maxsYu[3] = {2.,2.,8.};

  const theia::op_tensor_interpolation<DIM,FLT,VAL,0> info;
  int LL[DIM] = {L,L,L};
  theia::TensorP2M<DIM,VAL> P2M_x, P2M_y;
  theia::P2M(minsX, maxsX, Nx, X, LL, P2M_x, info);
  theia::P2M(minsY, maxsY, Ny, Y, LL, P2M_y, info);
  theia::Kron<DIM,VAL> M2M_x, M2M_y;
  theia::M2M(LL, minsXu, maxsXu, LL, minsX, maxsX, M2M_x, info);
  theia::M2M(LL, minsYu, maxsYu, LL, minsY, maxsY, M2M_y, info);
  VAL *M2L = nullptr;
  theia::M2L(minsXu, maxsXu, LL, minsYu, maxsYu, LL, Kernel, M2L, info);

  int Ld = theia::myintpow(L,DIM);
  VAL *tmp0 = new VAL[Ld], *tmp1 = new VAL[Ld];

  // a) Composed operators
  theia::TensorP2M<DIM,VAL> T_x, T_y;
  gemm(M2M_x, P2M_x, T_x);
  gemm(M2M_y, P2M_y, T_y);
  gemm (T_y, q, tmp0, 1);
  theia::gemm(1., M2L, tmp0, 0., tmp1, Ld, Ld, 1);
  gemTm(T_x, tmp1, a, 1);

  // b) Step by step
  gemm (P2M_y, q, tmp0, 1);
  gemm (M2M_y, tmp0, tmp1, 1);
  theia::gemm(1., M2L, tmp1, 0., tmp0, Ld, Ld, 1);
  gemTm(M2M_x, tmp0, tmp1, 1);
  gemTm(P2M_x, tmp1, b, 1);

  // Direct evaluation
  VAL* Mat = new VAL[Nx*Ny];
  Kernel(X,Nx,Y,Ny,Mat);
  theia::gemm(1.,Mat,q,0.,e,Nx,Ny,1);
  FLT errmax = 0., diffmax = 0.;
  for(int i = 0; i < Nx; i++){
    errmax  = std::max(errmax , std::abs(a[i]-e[i])/std::abs(e[i]));
    diffmax = std::max(diffmax, std::abs(a[i]-b[i])/std::abs(b[i]));
  }

  // Nestedness: M2M * P2M_child == P2M_parent
  theia::TensorP2M<DIM,VAL> P2M_yu;
  theia::P2M(minsYu, maxsYu, Ny, Y, LL, P2M_yu, info);
  VAL *D0 = nullptr, *D1 = nullptr;
  T_y.to_dense(D0);
  P2M_yu.to_dense(D1);
  FLT nest = 0.;
  for(int i = 0; i < Ld*Ny; i++){nest = std::max(nest, std::abs(D0[i]-D1[i]));}

  std::cout << std::boolalpha << (errmax < 1.e-8 && diffmax < 1.e-12 && nest < 1.e-12) << std::endl;

  delete [] X; delete [] Y; delete [] q; delete [] a; delete [] b; delete [] e; delete [] M2L;
  delete [] tmp0; delete [] tmp1; delete [] Mat; delete [] D0; delete [] D1;
  return 0;
}
