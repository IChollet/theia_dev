#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/theia.hpp"  

#define FLT double
#define VAL std::complex<FLT>
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

// P2M / L2P in tensor form (op_tensor_interpolation), compared to direct evaluation
// and to the dense P2M of op_interpolation
int main(){
  srand(235);
  int Nx = 124, Ny = 714, L = 12;
  std::array<FLT,DIM>* X = new std::array<FLT,DIM>[Nx];
  std::array<FLT,DIM>* Y = new std::array<FLT,DIM>[Ny];
  VAL *q = new VAL[Ny], *a = new VAL[Nx], *e = new VAL[Nx];
  light Kernel;
  FLT minsX[3] = {0.,0.,0.}, maxsX[3] = {1.,1.,1.};
  FLT minsY[3] = {0.,0.,2.}, maxsY[3] = {1.,1.,3.};
  for(int i = 0; i < Nx; i++){for(int k = 0; k < DIM; k++){X[i][k] = minsX[k] + urand*(maxsX[k]-minsX[k]);}}
  for(int i = 0; i < Ny; i++){for(int k = 0; k < DIM; k++){Y[i][k] = minsY[k] + urand*(maxsY[k]-minsY[k]);} q[i] = VAL(urand);}

  // Operators
  const theia::op_tensor_interpolation<DIM,FLT,VAL,0> info;
  int LL[DIM] = {L,L,L};
  theia::TensorP2M<DIM,VAL> P2M_x, P2M_y;
  VAL *M2L = nullptr;
  theia::P2M(minsX, maxsX, Nx, X, LL, P2M_x, info);
  theia::P2M(minsY, maxsY, Ny, Y, LL, P2M_y, info);
  theia::M2L(minsX, maxsX, LL, minsY, maxsY, LL, Kernel, M2L, info);

  // Apply: P2M, M2L, L2P (= P2M^T)
  int Ld = theia::myintpow(L,DIM);
  VAL *tmp0 = new VAL[Ld], *tmp1 = new VAL[Ld];
  gemm (P2M_y, q, tmp0, 1);
  theia::gemm(1., M2L, tmp0, 0., tmp1, Ld, Ld, 1);
  gemTm(P2M_x, tmp1, a, 1);

  // Direct evaluation
  VAL* Mat = new VAL[Nx*Ny];
  Kernel(X,Nx,Y,Ny,Mat);
  theia::gemm(1.,Mat,q,0.,e,Nx,Ny,1);
  FLT errmax = 0.;
  for(int i = 0; i < Nx; i++){errmax = std::max(errmax, std::abs(a[i]-e[i])/std::abs(e[i]));}

  // Same matrix as the dense P2M
  VAL *Sd = nullptr, *St = nullptr;
  theia::P2M(minsY, maxsY, Ny, Y, LL, Sd, theia::op_interpolation<DIM,FLT,VAL,0>());
  P2M_y.to_dense(St);
  FLT dmax = 0.;
  for(int i = 0; i < Ld*Ny; i++){dmax = std::max(dmax, std::abs(Sd[i]-St[i]));}

  std::cout << std::boolalpha << (errmax < 1.e-8 && dmax < 1.e-14) << std::endl;

  delete [] X; delete [] Y; delete [] q; delete [] a; delete [] e; delete [] M2L;
  delete [] tmp0; delete [] tmp1; delete [] Mat; delete [] Sd; delete [] St;
  return 0;
}
