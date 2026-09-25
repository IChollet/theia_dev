#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/theia.hpp"  

#define FLT double
#define DIM 3
#define urand rand()/FLT(RAND_MAX)

class light{
public :
  light(){}
  void operator()(std::array<FLT,DIM>* X, int Nx, std::array<FLT,DIM>* Y, int Ny, FLT* A){
    for(int j = 0; j < Ny; j++){
      for(int i = 0; i < Nx; i++){
        FLT R0 = X[i][0] - Y[j][0];
	FLT R1 = X[i][1] - Y[j][1];
	FLT R2 = X[i][2] - Y[j][2];
	FLT k  = .5;
	FLT R  = R0*R0 + R1*R1 + R2*R2;
	FLT r  = sqrt(R);
	A[j*Nx+i] = k * exp(-k*r) / R;
	if(std::isnan(A[j*Nx+i])){A[j*Nx+i] = 0.;}
      }
    }
  }
  
};

int main(int argc, char* argv[]){
  srand(235);
  
  // Parameters
  int   Nx = 374;
  int   Ny = 93;
  int   L  = 12;

  // Global data
  std::array<FLT,DIM>*  X = new std::array<FLT,DIM>[Nx];
  std::array<FLT,DIM>*  Y = new std::array<FLT,DIM>[Ny];
  FLT*  q = new FLT[Ny];
  FLT*  a = new FLT[Nx];
  FLT*  e = new FLT[Nx];
  light Kernel;

  // Leaf clusters and sampling
  FLT minsX[3]; minsX[0] = 0.;  minsX[1] = 0.; minsX[2] = 0.;
  FLT maxsX[3]; maxsX[0] = 1.;  maxsX[1] = 1.; maxsX[2] = 1.;
  FLT minsY[3]; minsY[0] = 0.;  minsY[1] = 0.; minsY[2] = 6.;
  FLT maxsY[3]; maxsY[0] = 1.;  maxsY[1] = 1.; maxsY[2] = 7.;
  for(int i = 0; i < Nx; i++){
    for(int k = 0; k < DIM; k++){
      FLT xx = minsX[k] + urand * (maxsX[k] - minsX[k]);
      X[i][k] = xx;
    }
  }
  for(int i = 0; i < Ny; i++){
    for(int k = 0; k < DIM; k++){
      FLT yy = minsY[k] + urand * (maxsY[k] - minsY[k]);
      Y[i][k] = yy;
    }
    q[i] = urand;
  }

  // Non-leaf clusters
  FLT minsXu[3]; minsXu[0] = 0.;  minsXu[1] = 0.; minsXu[2] = 0.;
  FLT maxsXu[3]; maxsXu[0] = 2.;  maxsXu[1] = 2.; maxsXu[2] = 2.;
  FLT minsYu[3]; minsYu[0] = 0.;  minsYu[1] = 0.; minsYu[2] = 6.;
  FLT maxsYu[3]; maxsYu[0] = 2.;  maxsYu[1] = 2.; maxsYu[2] = 8.;
  
  // Operator infos
  const theia::op_interpolation<DIM,FLT,FLT,0> info;
  int LL[DIM] = {L,L,L};
  
  // P2M matrices
  FLT *P2M_x = nullptr;
  FLT *P2M_y = nullptr;
  theia::P2M(minsX, maxsX, Nx, X, LL, P2M_x, info);
  theia::P2M(minsY, maxsY, Ny, Y, LL, P2M_y, info);

  // M2M matrices
  theia::Kron<DIM,FLT> M2M_x;
  theia::Kron<DIM,FLT> M2M_y;
  theia::M2M(LL, minsXu, maxsXu, LL, minsX, maxsX, M2M_x, info);
  theia::M2M(LL, minsYu, maxsYu, LL, minsY, maxsY, M2M_y, info);
  
  // M2L matrices
  FLT *M2L = nullptr;
  theia::M2L(minsXu, maxsXu, LL, minsYu, maxsYu, LL, Kernel, M2L, info);

  // Apply matrices
  int Ld = theia::myintpow(L,DIM);
  FLT *tmp0 = new FLT[Ld];
  FLT *tmp1 = new FLT[Ld];
  theia::gemm (1., P2M_y, q   , 0., tmp0, Ld ,Ny, 1);
  gemm (M2M_y, tmp0, tmp1, 1);
  theia::gemm (1., M2L  , tmp1, 0., tmp0, Ld ,Ld, 1);
  gemTm(M2M_x, tmp0, tmp1, 1);
  theia::gemTm(1., P2M_x, tmp1, 0., a   , Nx ,Ld, 1);
  
  // Tests and output
  FLT Mat[Nx*Ny];
  Kernel(X,Nx,Y,Ny,Mat);
  theia::gemm(1.,Mat,q,0.,e,Nx,Ny,1);
  FLT errmax = 0.;
  for(int i = 0; i < Nx; i++){
    FLT loc_err = std::abs(a[i]-e[i])/std::abs(e[i]);
    if(loc_err > errmax){errmax = loc_err;}
  }

  std::cout << std::boolalpha << (errmax < 1.e-8) << std::endl;

  return 0;
}
