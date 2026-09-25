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
  Reinterpolation with anisotropic orders, different per dimension, per level
  and per cluster: P2M (child) -> M2M -> M2L (parents) -> L2L -> L2P (child).
  Dense chain and composed tensor chain (M2M * P2M as a TensorP2M), and
  nestedness M2M * P2M_child == P2M_parent (parent orders <= child orders).
*/
int main(){
  srand(235);
  int Nx = 213, Ny = 341;
  int LCx[DIM] = {6,11,15}, LPx[DIM] = {5,10,14};   // child / parent orders, targets
  int LCy[DIM] = {7,11,14}, LPy[DIM] = {6,11,13};   // child / parent orders, sources
  std::array<FLT,DIM>* X = new std::array<FLT,DIM>[Nx];
  std::array<FLT,DIM>* Y = new std::array<FLT,DIM>[Ny];
  VAL *q = new VAL[Ny], *a = new VAL[Nx], *b = new VAL[Nx], *e = new VAL[Nx];
  light Kernel;

  // Leaf (child) and parent clusters, thin along x and long along z
  FLT minsX [3] = {0.,0.,0.}, maxsX [3] = {0.25,1.,2.};
  FLT minsY [3] = {4.,0.,0.}, maxsY [3] = {4.25,1.,2.};
  FLT minsXu[3] = {0.,0.,0.}, maxsXu[3] = {0.5 ,2.,4.};
  FLT minsYu[3] = {4.,0.,0.}, maxsYu[3] = {4.5 ,2.,4.};
  for(int i = 0; i < Nx; i++){for(int k = 0; k < DIM; k++){X[i][k] = minsX[k] + urand*(maxsX[k]-minsX[k]);}}
  for(int i = 0; i < Ny; i++){for(int k = 0; k < DIM; k++){Y[i][k] = minsY[k] + urand*(maxsY[k]-minsY[k]);} q[i] = VAL(urand);}
  int LCxd = theia::prod_of_orders(LCx,DIM), LPxd = theia::prod_of_orders(LPx,DIM);
  int LCyd = theia::prod_of_orders(LCy,DIM), LPyd = theia::prod_of_orders(LPy,DIM);
  int Lmax = std::max(std::max(LCxd,LPxd),std::max(LCyd,LPyd));
  VAL *tmp0 = new VAL[Lmax], *tmp1 = new VAL[Lmax];

  // Common operators
  const theia::op_interpolation       <DIM,FLT,VAL,0> info;
  const theia::op_tensor_interpolation<DIM,FLT,VAL,0> tinfo;
  theia::Kron<DIM,VAL> M2M_x, M2M_y;
  theia::M2M(LPx, minsXu, maxsXu, LCx, minsX, maxsX, M2M_x, info);
  theia::M2M(LPy, minsYu, maxsYu, LCy, minsY, maxsY, M2M_y, info);
  VAL *M2L = nullptr;
  theia::M2L(minsXu, maxsXu, LPx, minsYu, maxsYu, LPy, Kernel, M2L, info);

  // a) Dense chain
  VAL *P2M_x = nullptr, *P2M_y = nullptr;
  theia::P2M(minsX, maxsX, Nx, X, LCx, P2M_x, info);
  theia::P2M(minsY, maxsY, Ny, Y, LCy, P2M_y, info);
  theia::gemm (1., P2M_y, q, 0., tmp0, LCyd, Ny, 1);
  gemm (M2M_y, tmp0, tmp1, 1);
  theia::gemm (1., M2L, tmp1, 0., tmp0, LPxd, LPyd, 1);
  gemTm(M2M_x, tmp0, tmp1, 1);
  theia::gemTm(1., P2M_x, tmp1, 0., a, Nx, LCxd, 1);

  // b) Tensor chain with composed operators M2M * P2M
  theia::TensorP2M<DIM,VAL> T_x, T_y;
  theia::P2M(minsX, maxsX, Nx, X, LCx, T_x, tinfo);
  theia::P2M(minsY, maxsY, Ny, Y, LCy, T_y, tinfo);
  gemm(M2M_x, T_x, T_x);
  gemm(M2M_y, T_y, T_y);
  gemm (T_y, q, tmp0, 1);
  theia::gemm(1., M2L, tmp0, 0., tmp1, LPxd, LPyd, 1);
  gemTm(T_x, tmp1, b, 1);

  // Direct evaluation
  VAL* Mat = new VAL[Nx*Ny];
  Kernel(X,Nx,Y,Ny,Mat);
  theia::gemm(1.,Mat,q,0.,e,Nx,Ny,1);
  FLT erra = 0., diff = 0.;
  for(int i = 0; i < Nx; i++){
    erra = std::max(erra, std::abs(a[i]-e[i])/std::abs(e[i]));
    diff = std::max(diff, std::abs(b[i]-a[i])/std::abs(a[i]));
  }

  // Nestedness
  VAL *D0 = nullptr, *D1 = nullptr;
  T_y.to_dense(D0);
  theia::P2M(minsYu, maxsYu, Ny, Y, LPy, D1, info);
  FLT nest = 0.;
  for(int i = 0; i < LPyd*Ny; i++){nest = std::max(nest, std::abs(D0[i]-D1[i]));}

  std::cout << std::boolalpha << (erra < 1.e-5 && diff < 1.e-12 && nest < 1.e-12) << std::endl;

  delete [] X; delete [] Y; delete [] q; delete [] a; delete [] b; delete [] e; delete [] tmp0; delete [] tmp1;
  delete [] M2L; delete [] P2M_x; delete [] P2M_y; delete [] Mat; delete [] D0; delete [] D1;
  return 0;
}
