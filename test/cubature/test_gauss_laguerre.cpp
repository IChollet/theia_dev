#include <iostream>
#include "../../include/cubature/cubature.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <complex>

#define FLT double
#define DIM 3

int main(){
  constexpr int    dim     = DIM;
  constexpr int    Ntest   = 10000;
  constexpr int    order   = 7;
  constexpr FLT    scaling = 0.5;
  std::array<std::complex<FLT>,DIM> b;

  // Fonction f(x) = e^{<b,x>}
  auto integrand = [&](const std::array<FLT,DIM>& x) -> std::complex<FLT>{
    std::complex<FLT> dot = 0.;
    for(int i = 0; i < dim; ++i) dot += b[i] * x[i];
    return std::exp(dot);
  };
  
  auto rule    = theia::cubature::gauss_laguerre::get<FLT>(order);
  auto ruleDIM = theia::cubature::tensor_rule<FLT,1,FLT,FLT,DIM>(rule);

  FLT num = 0.;
  FLT den = 0.;
  for(int jj = 0; jj < Ntest; ++jj){
    for(int i = 0; i < dim; ++i){
      FLT real = scaling * std::rand() / FLT(RAND_MAX);
      FLT imag = scaling * (std::rand() / FLT(RAND_MAX) > 0.5 ? 1. : -1.) * std::rand() / FLT(RAND_MAX);
      b[i] = std::complex<FLT>(real,imag);
    }
    std::complex<FLT> numerical;
    eval(ruleDIM,integrand,numerical);
    std::complex<FLT> exact = 1.;
    for(int k = 0; k < dim; k++){
      exact *= 1./(1.-b[k]);
    }
    num += std::abs(numerical - exact);
    den += std::abs(exact);
  }

  std::cout << std::boolalpha << (num/den < 1.e-5) << std::endl;

  return 0;

}
