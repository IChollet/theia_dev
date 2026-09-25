#include <iomanip>
#include <cmath>
#include <iostream>
#include "../../include/theia.hpp"  

#define FLT double
#define DIM 3
#define urand rand()/FLT(RAND_MAX)

class light{
public :
  void operator()(std::array<FLT,DIM>* X, int Nx, std::array<FLT,DIM>* Y, int Ny, FLT* A){
    for(int j = 0; j < Ny; j++){
      for(int i = 0; i < Nx; i++){
        FLT R0 = X[i][0] - Y[j][0];
	FLT R1 = X[i][1] - Y[j][1];
	FLT R2 = X[i][2] - Y[j][2];
	FLT R  = R0*R0 + R1*R1 + R2*R2;
	A[j*Nx+i] = .5 * exp(-.5*sqrt(R)) / R;
      }
    }
  }
};

// P2M / M2L / L2P with different interpolation orders on targets (Lx) and sources (Ly)
int main(){
  srand(235);
  int Nx = 124, Ny = 214, Lx = 10, Ly = 12;
  std::array<FLT,DIM>* X = new std::array<FLT,DIM>[Nx];
  std::array<FLT,DIM>* Y = new std::array<FLT,DIM>[Ny];
  FLT *q = new FLT[Ny], *a = new FLT[Nx], *e = new FLT[Nx];
  light Kernel;
  FLT minsX[3] = {0.,0.,0.}, maxsX[3] = {1.,1.,1.};
  FLT minsY[3] = {0.,0.,2.}, maxsY[3] = {1.,1.,3.};
  for(int i = 0; i < Nx; i++){for(int k = 0; k < DIM; k++){X[i][k] = minsX[k] + urand*(maxsX[k]-minsX[k]);}}
  for(int i = 0; i < Ny; i++){for(int k = 0; k < DIM; k++){Y[i][k] = minsY[k] + urand*(maxsY[k]-minsY[k]);} q[i] = urand;}

  const theia::op_interpolation<DIM,FLT,FLT,0> info;
  int LX[DIM] = {Lx,Lx,Lx}, LY[DIM] = {Ly,Ly,Ly};
  FLT *P2M_x = nullptr, *P2M_y = nullptr, *M2L = nullptr;
  theia::P2M(minsX, maxsX, Nx, X, LX, P2M_x, info);
  theia::P2M(minsY, maxsY, Ny, Y, LY, P2M_y, info);
  theia::M2L(minsX, maxsX, LX, minsY, maxsY, LY, Kernel, M2L, info);

  int Lxd = theia::myintpow(Lx,DIM), Lyd = theia::myintpow(Ly,DIM);
  FLT *tmp0 = new FLT[Lyd], *tmp1 = new FLT[Lxd];
  theia::gemm (1., P2M_y, q   , 0., tmp0, Lyd, Ny , 1);
  theia::gemm (1., M2L  , tmp0, 0., tmp1, Lxd, Lyd, 1);
  theia::gemTm(1., P2M_x, tmp1, 0., a   , Nx , Lxd, 1);

  FLT* Mat = new FLT[Nx*Ny];
  Kernel(X,Nx,Y,Ny,Mat);
  theia::gemm(1.,Mat,q,0.,e,Nx,Ny,1);
  FLT errmax = 0.;
  for(int i = 0; i < Nx; i++){errmax = std::max(errmax, std::abs(a[i]-e[i])/std::abs(e[i]));}
  std::cout << std::boolalpha << (errmax < 1.e-6) << std::endl;

  delete [] X; delete [] Y; delete [] q; delete [] a; delete [] e;
  delete [] P2M_x; delete [] P2M_y; delete [] M2L; delete [] tmp0; delete [] tmp1; delete [] Mat;
  return 0;
}
