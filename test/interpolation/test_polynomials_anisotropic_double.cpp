#include <iomanip>
#include <cmath>
#include <iostream>
#include "../../include/theia.hpp"  

#define FLT double
#define DIM 3
#define urand rand()/FLT(RAND_MAX)

/*
  Per-dimension orders for nodes and polynomials:
   - same orders everywhere: identical to the scalar-order version
   - anisotropic orders: Kronecker product of the 1D polynomials, and
     nodes of the grid = 1D nodes of each dimension (last dimension fastest)
*/
int main(){
  srand(235);
  const int N = 57;
  FLT mins[DIM] = {-1.,0.5,2.}, maxs[DIM] = {0.,1.5,5.};
  std::array<FLT,DIM>* p = new std::array<FLT,DIM>[N];
  for(int j = 0; j < N; j++){for(int k = 0; k < DIM; k++){p[j][k] = mins[k] + urand*(maxs[k]-mins[k]);}}
  bool ok = true;

  // Isotropic: array version == scalar version (up to rounding)
  int L = 7, LL[DIM] = {7,7,7}, Ld = L*L*L;
  FLT *S0 = nullptr, *S1 = nullptr;
  theia::get_polynomials<DIM,FLT,FLT,0>(L , S0, mins, maxs, p, N);
  theia::get_polynomials<DIM,FLT,FLT,0>(LL, S1, mins, maxs, p, N);
  for(int i = 0; i < Ld*N; i++){ok &= (std::abs(S0[i]-S1[i]) < 1.e-14);}
  std::array<FLT,DIM> *z0 = new std::array<FLT,DIM>[Ld], *z1 = new std::array<FLT,DIM>[Ld];
  theia::get_multivariate_interp_nodes<DIM,FLT,0>(L , mins, maxs, z0);
  theia::get_multivariate_interp_nodes<DIM,FLT,0>(LL, mins, maxs, z1);
  for(int i = 0; i < Ld; i++){for(int k = 0; k < DIM; k++){ok &= (std::abs(z0[i][k]-z1[i][k]) < 1.e-14);}}

  // Anisotropic polynomials: S(:,j) = S_0(:,j) x S_1(:,j) x S_2(:,j)
  int LA[DIM] = {4,9,6}, LAd = 4*9*6;
  FLT *SA = nullptr, *F[DIM];
  theia::get_polynomials<DIM,FLT,FLT,0>(LA, SA, mins, maxs, p, N);
  std::array<FLT,1>* pk = new std::array<FLT,1>[N];
  for(int k = 0; k < DIM; k++){
    F[k] = nullptr;
    for(int j = 0; j < N; j++){pk[j][0] = p[j][k];}
    theia::get_polynomials<1,FLT,FLT,0>(LA[k], F[k], mins+k, maxs+k, pk, N);
  }
  FLT err = 0.;
  for(int j = 0; j < N; j++){
    FLT sum = 0.;
    for(int i0 = 0; i0 < LA[0]; i0++){
      for(int i1 = 0; i1 < LA[1]; i1++){
	for(int i2 = 0; i2 < LA[2]; i2++){
	  FLT ref = F[0][j*LA[0]+i0] * F[1][j*LA[1]+i1] * F[2][j*LA[2]+i2];
	  FLT val = SA[j*LAd + (i0*LA[1]+i1)*LA[2]+i2];
	  err  = std::max(err, std::abs(val-ref));
	  sum += val;
	}
      }
    }
    ok &= (std::abs(sum-1.) < 1.e-12);   // partition of unity
  }
  ok &= (err < 1.e-14);

  // Anisotropic nodes
  std::array<FLT,DIM>* zA = new std::array<FLT,DIM>[LAd];
  theia::get_multivariate_interp_nodes<DIM,FLT,0>(LA, mins, maxs, zA);
  for(int i0 = 0; i0 < LA[0]; i0++){
    for(int i1 = 0; i1 < LA[1]; i1++){
      for(int i2 = 0; i2 < LA[2]; i2++){
	std::array<FLT,DIM>& z = zA[(i0*LA[1]+i1)*LA[2]+i2];
	ok &= (std::abs(z[0] - theia::get_node<0>(i0,LA[0],mins[0],maxs[0])) < 1.e-14);
	ok &= (std::abs(z[1] - theia::get_node<0>(i1,LA[1],mins[1],maxs[1])) < 1.e-14);
	ok &= (std::abs(z[2] - theia::get_node<0>(i2,LA[2],mins[2],maxs[2])) < 1.e-14);
      }
    }
  }

  std::cout << std::boolalpha << ok << std::endl;

  delete [] p; delete [] S0; delete [] S1; delete [] z0; delete [] z1; delete [] SA; delete [] pk; delete [] zA;
  for(int k = 0; k < DIM; k++){delete [] F[k];}
  return 0;
}
