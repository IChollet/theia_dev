#include <iomanip>
#include <cmath>
#include <iostream>
#include "../../include/theia.hpp"  

#define FLT double
#define urand rand()/FLT(RAND_MAX)

// Chebyshev Lagrange polynomials must be finite and sum to 1 for:
//  - particles lying exactly on the bounds of a (tight) box
//  - flat boxes (mins == maxs along one dimension)
int main(){
  srand(235);
  const int L = 8;
  bool ok = true;

  // Particles on the box bounds (1D, random boxes)
  for(int t = 0; t < 10000 && ok; t++){
    FLT a = urand*10.-5., b = a + urand*3. + 1.e-3;
    FLT mins[1] = {a}, maxs[1] = {b};
    std::array<FLT,1> p[2] = {{a},{b}};
    FLT* S = nullptr;
    theia::get_polynomials<1,FLT,FLT,0>(L,S,mins,maxs,p,2);
    for(int j = 0; j < 2; j++){
      FLT sum = 0.;
      for(int i = 0; i < L; i++){ok &= std::isfinite(S[j*L+i]); sum += S[j*L+i];}
      ok &= (std::abs(sum-1.) < 1.e-12);
    }
    delete [] S;
  }

  // Flat box along the last dimension (2D)
  FLT mins[2] = {0.,0.5}, maxs[2] = {1.,0.5};
  std::array<FLT,2> p[1] = {{0.3,0.5}};
  FLT* S = nullptr;
  theia::get_polynomials<2,FLT,FLT,0>(L,S,mins,maxs,p,1);
  FLT sum = 0.;
  for(int i = 0; i < L*L; i++){ok &= std::isfinite(S[i]); sum += S[i];}
  ok &= (std::abs(sum-1.) < 1.e-12);
  delete [] S;

  std::cout << std::boolalpha << ok << std::endl;
  return 0;
}
