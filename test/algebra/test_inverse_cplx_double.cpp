#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/theia.hpp"  

#define urand rand()/double(RAND_MAX)
// FLT est la précision à utiliser
// MTX est le type de retour du noyau que l'on considère
#define FLT double
#define MTX std::complex<FLT>

class light{
public :
  light(){}
  void operator()(std::array<FLT,3>* X, int Nx, std::array<FLT,3>* Y, int Ny, MTX* A){
    for(int j = 0; j < Ny; j++){
      for(int i = 0; i < Nx; i++){
        FLT R0 = X[i][0] - Y[j][0];
	FLT R1 = X[i][1] - Y[j][1];
	FLT R2 = X[i][2] - Y[j][2];
	FLT k  = .5;
	FLT R  = R0*R0 + R1*R1 + R2*R2;
	FLT r  = sqrt(R);
        A[j*Nx+i] = 1. / R;
      }
    }
  }
};


int main(int argc, char* argv[]){

  int   Nx   = 5;
  int   Ny   = Nx;
  std::array<FLT,3>*  X    = new std::array<FLT,3>[Nx];
  std::array<FLT,3>*  Y    = new std::array<FLT,3>[Ny];
  light Kernel;
  FLT minsX[3]; minsX[0] = 0.;  minsX[1] = 0.; minsX[2] = 0.;
  FLT maxsX[3]; maxsX[0] = 1.;  maxsX[1] = 1.; maxsX[2] = 1.;
  FLT minsY[3]; minsY[0] = 0.;  minsY[1] = 0.; minsY[2] = 2.;
  FLT maxsY[3]; maxsY[0] = 1.;  maxsY[1] = 1.; maxsY[2] = 3.;
  for(int i = 0; i < Nx; i++){
    for(int k = 0; k < 3; k++){
      FLT xx = minsX[k] + urand * (maxsX[k] - minsX[k]);
      X[i][k] = xx;
    }
  }
  for(int i = 0; i < Ny; i++){
    for(int k = 0; k < 3; k++){
      FLT yy = minsY[k] + urand * (maxsY[k] - minsY[k]);
      Y[i][k] = yy;
    }
  }

  MTX A[Nx*Ny], B[Nx*Ny], C[Nx*Ny];
  Kernel(X,Nx,Y,Ny,A);
  Kernel(X,Nx,Y,Ny,B);
  theia::invert(A,Nx);
  theia::gemm(1.,A,B,0.,C,Nx,Nx,Nx);
  MTX err = 0.;
  for(int i = 0; i < Nx; i++){
    for(int j = 0; j < Nx; j++){
      err += (i==j ? (C[i+i*Nx] - MTX(1.)) : C[i+j*Nx]);
    }
  }
  std::cout << std::boolalpha << (std::abs(err) < 1.e-3) << std::endl;

  return 0;
}
