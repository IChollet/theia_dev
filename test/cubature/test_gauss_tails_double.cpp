#include <iomanip>
#include <cmath>
#include <iostream>
#include "../../include/cubature/cubature.hpp"

#define FLT double

using namespace theia::cubature;

/*
  Gauss-Hermite / Gauss-Laguerre with integrands concentrated in the tails,
  where the weights are tiny (down to 1e-128): they must keep a full relative
  precision (Newton refinement + Christoffel weights).
   - Laguerre (weight e^{-x})      : int x^k e^{-x} / k! = 1            (exact for k <= 2n-1)
   - Hermite  (weight e^{-x^2/2})  : int e^{bx} e^{-x^2/2} = sqrt(2 pi) e^{b^2/2}
*/
int main(){
  bool ok = true;

  for(int n : {20, 40, 80}){
    auto r = gauss_laguerre::get<FLT>(n);
    for(int k : {0, n, 2*n-1}){
      // sum_i w_i x_i^k / k! = 1, evaluated with logarithms (x^k and k! overflow)
      FLT s = 0.;
      for(int i = 0; i < r.N; i++){s += r.w[i] * std::exp(k*std::log(r.x[i][0]) - std::lgamma(k+1.));}
      ok &= (std::abs(s-1.) < 1.e-13);
    }
    for(int i = 0; i < r.N; i++){ok &= (r.w[i] > 0.);}
  }

  for(int n : {60, 120, 300}){
    auto r = gauss_hermite::get<FLT>(n);
    for(FLT b : {0., 4., 8.}){
      FLT s = 0.;
      for(int i = 0; i < r.N; i++){s += r.w[i] * std::exp(b*r.x[i][0]);}
      FLT exact = std::sqrt(2.*M_PI) * std::exp(b*b/2.);
      ok &= (std::abs(s-exact)/exact < 1.e-13);
    }
    for(int i = 0; i < r.N; i++){ok &= (r.w[i] > 0.);}
  }

  std::cout << std::boolalpha << ok << std::endl;
  return 0;
}
