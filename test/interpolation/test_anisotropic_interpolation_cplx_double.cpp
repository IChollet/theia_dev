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

/*
  Anisotropic orders on flattened clusters (thin along x, long along z):
  P2M / M2L / L2P with different orders per dimension and per cluster,
  with the dense (op_interpolation) and tensor (op_tensor_interpolation) operators.
*/
int main(){
  srand(235);
  int Nx = 213, Ny = 341;
  int LX[DIM] = {6,11,15};
  int LY[DIM] = {5,11,16};
  std::array<FLT,DIM>* X = new std::array<FLT,DIM>[Nx];
  std::array<FLT,DIM>* Y = new std::array<FLT,DIM>[Ny];
  VAL *q = new VAL[Ny], *a = new VAL[Nx], *b = new VAL[Nx], *e = new VAL[Nx];
  light Kernel;
  FLT minsX[3] = {0.,0.,0.}, maxsX[3] = {0.25,1.,2.};
  FLT minsY[3] = {2.,0.,0.}, maxsY[3] = {2.25,1.,2.};
  for(int i = 0; i < Nx; i++){for(int k = 0; k < DIM; k++){X[i][k] = minsX[k] + urand*(maxsX[k]-minsX[k]);}}
  for(int i = 0; i < Ny; i++){for(int k = 0; k < DIM; k++){Y[i][k] = minsY[k] + urand*(maxsY[k]-minsY[k]);} q[i] = VAL(urand);}
  int Lxd = theia::prod_of_orders(LX,DIM), Lyd = theia::prod_of_orders(LY,DIM);
  VAL *tmp0 = new VAL[Lyd], *tmp1 = new VAL[Lxd];

  // Dense operators
  const theia::op_interpolation<DIM,FLT,VAL,0> info;
  VAL *P2M_x = nullptr, *P2M_y = nullptr, *M2L = nullptr;
  theia::P2M(minsX, maxsX, Nx, X, LX, P2M_x, info);
  theia::P2M(minsY, maxsY, Ny, Y, LY, P2M_y, info);
  theia::M2L(minsX, maxsX, LX, minsY, maxsY, LY, Kernel, M2L, info);
  theia::gemm (1., P2M_y, q   , 0., tmp0, Lyd, Ny , 1);
  theia::gemm (1., M2L  , tmp0, 0., tmp1, Lxd, Lyd, 1);
  theia::gemTm(1., P2M_x, tmp1, 0., a   , Nx , Lxd, 1);

  // Tensor operators
  const theia::op_tensor_interpolation<DIM,FLT,VAL,0> tinfo;
  theia::TensorP2M<DIM,VAL> T_x, T_y;
  theia::P2M(minsX, maxsX, Nx, X, LX, T_x, tinfo);
  theia::P2M(minsY, maxsY, Ny, Y, LY, T_y, tinfo);
  gemm (T_y, q, tmp0, 1);
  theia::gemm(1., M2L, tmp0, 0., tmp1, Lxd, Lyd, 1);
  gemTm(T_x, tmp1, b, 1);

  // Direct evaluation
  VAL* Mat = new VAL[Nx*Ny];
  Kernel(X,Nx,Y,Ny,Mat);
  theia::gemm(1.,Mat,q,0.,e,Nx,Ny,1);
  FLT erra = 0., errb = 0.;
  for(int i = 0; i < Nx; i++){
    erra = std::max(erra, std::abs(a[i]-e[i])/std::abs(e[i]));
    errb = std::max(errb, std::abs(b[i]-e[i])/std::abs(e[i]));
  }

  // Tensor and dense P2M are the same matrix
  VAL* D = nullptr;
  T_y.to_dense(D);
  FLT dmax = 0.;
  for(int i = 0; i < Lyd*Ny; i++){dmax = std::max(dmax, std::abs(D[i]-P2M_y[i]));}

  std::cout << std::boolalpha << (erra < 1.e-7 && errb < 1.e-7 && dmax < 1.e-14) << std::endl;

  delete [] X; delete [] Y; delete [] q; delete [] a; delete [] b; delete [] e;
  delete [] tmp0; delete [] tmp1; delete [] P2M_x; delete [] P2M_y; delete [] M2L; delete [] Mat; delete [] D;
  return 0;
}
