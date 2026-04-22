#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../include/theia.hpp"  

#define urand rand()/double(RAND_MAX)

class light{
public :
  light(){}
  void operator()(std::array<double,3>* X, int Nx, std::array<double,3>* Y, int Ny, double* A){
    for(int j = 0; j < Ny; j++){
      for(int i = 0; i < Nx; i++){
        double R0 = X[i][0] - Y[j][0];
	double R1 = X[i][1] - Y[j][1];
	double R2 = X[i][2] - Y[j][2];
	double k  = .5;
	double R  = R0*R0 + R1*R1 + R2*R2;
	double r  = sqrt(R);
        A[j*Nx+i] = k * exp(-k*r) / R;
	if(std::isnan(A[j*Nx+i])){A[j*Nx+i] = 0.;}
      }
    }
  }
};

int main(int argc, char* argv[]){

  // Parameters
  int   Nx   = 428;
  int   Ny   = 781;
  int   L    = 7;
  std::array<double,3>*  X    = new std::array<double,3>[Nx];
  std::array<double,3>*  Y    = new std::array<double,3>[Ny];
  double*  q    = new double[Ny];
  double*  a    = new double[Nx];
  double*  e    = new double[Nx];
  light Kernel;
  double minsX[3]; minsX[0] = 0.;  minsX[1] = 0.; minsX[2] = 0.;
  double maxsX[3]; maxsX[0] = 1.;  maxsX[1] = 1.; maxsX[2] = 1.;
  double minsY[3]; minsY[0] = 0.;  minsY[1] = 0.; minsY[2] = 2.;
  double maxsY[3]; maxsY[0] = 1.;  maxsY[1] = 1.; maxsY[2] = 3.;
  for(int i = 0; i < Nx; i++){
    for(int k = 0; k < 3; k++){
      double xx = minsX[k] + urand * (maxsX[k] - minsX[k]);
      X[i][k] = xx;
    }
  }
  for(int i = 0; i < Ny; i++){
    for(int k = 0; k < 3; k++){
      double yy = minsY[k] + urand * (maxsY[k] - minsY[k]);
      Y[i][k] = yy;
    }
    q[i] = urand;
  }


  // Lagrange Interpolation for Target and Sources (LITS)
  // a) Declare interpolation matrices on two clusters
  //    1) templates
  //      type for kernel values (i.e. output type for kernel evaluations)
  //      floating point precision
  //      dimension of the problem
  //      kernel class
  //      0 for chebyshev / 1 for equispaced
  //    2) arguments
  //      minsX / maxsX  --> bounds for target cluster
  //      X              --> set of target particles
  //      Nx             --> number of target paticles
  //      minsY / maxsY  --> bounds for source cluster
  //      X              --> set of source particles
  //      Nx             --> number of source paticles
  //      L              --> 1D interpolation order
  //      Kernel         --> user defined kernel
  theia::lits<double,
	      double,
	      3,
	      light,
	      0>
    GL(minsX,maxsX,X,Nx,
       minsY,maxsY,Y,Ny,
       L, &Kernel);
  // b) Precompute the low-rank version (SVD precision as input)
  GL.get_UV(1.e-7);
  // c) Apply the low-rank approximation "GL" to a vector "q"
  gemm(GL,q,a,1);

  // Tests and output
  std::cout << "Rank of interpolated matrix: " << Rank(GL) << std::endl;
  double Mat[Nx*Ny];
  Kernel(X,Nx,Y,Ny,Mat);
  theia::gemm(1.,Mat,q,0.,e,Nx,Ny,1);
  double errmax = 0.;
  for(int i = 0; i < Nx; i++){
    double loc_err = std::abs(a[i]-e[i])/std::abs(e[i]);
    if(loc_err > errmax){errmax = loc_err;}
  }
  std::cout << "Error: " << errmax << std::endl;

  // Test on second prototype
  double *U, *V; int rank;
  /*
  theia::get_lits_cheb<double,
	      double,
	      3,
	      light>(minsX,maxsX,X,Nx,
		     minsY,maxsY,Y,Ny,
		     L, &Kernel,
		     1.e-7, U, V, rank);
  */
  int frank = 50;
  theia::get_lits_cheb_fixed_rank<double,
				  double,
				  3,
				  light>(minsX,maxsX,X,Nx,
					 minsY,maxsY,Y,Ny,
					 L, &Kernel,
					 1.e-7, U, V, frank);
  //GL.get_UV(1.e-7, U, V, rank);
  double* tmp0 = new double[frank];
  theia::gemm(1.,V,q,0.,tmp0,frank,Ny, 1);
  theia::gemm(1.,U,tmp0,0.,a,Nx, frank,1);
  errmax = 0.;
  for(int i = 0; i < Nx; i++){
    double loc_err = std::abs(a[i]-e[i])/std::abs(e[i]);
    if(loc_err > errmax){errmax = loc_err;}
  }
  std::cout << "Error (function-like prototype): " << errmax << std::endl;

  return 0;
}
