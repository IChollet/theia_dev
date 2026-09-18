#include <iostream>
#include "../../include/cubature/cubature.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>

#define FLT double
#define DIM 1


std::complex<FLT> f(const std::array<std::complex<FLT>,DIM>& z) {
  std::complex<FLT> s = 0.;
  for(int k = 0; k < DIM; k++){
    s += z[k]*z[k];
  }
  return std::complex<FLT>(1.)/std::sqrt(s);
}

int main(){
  constexpr int     dim     = DIM;
  constexpr int     Ntest   = 10;
  constexpr int     order   = 20;
  constexpr FLT     scaling = .25;
  std::complex<FLT> center  = std::complex<FLT>(8.,0.);
  FLT               radius  = 0.5;
  std::array<std::complex<FLT>,DIM> b;
  
  auto rule  = theia::cubature::cauchy::get<FLT>(order, center, radius);
  //print(rule);
  //auto rule2 = theia::cubature::tensor_rule(rule,rule);

  FLT num = 0.;
  FLT den = 0.;
  for(int jj = 0; jj < Ntest; ++jj){
    for(int i = 0; i < dim; ++i){
      b[i] = center + scaling * (std::rand() / FLT(RAND_MAX) > 0.5 ? 1. : -1.)
	* std::rand() / FLT(RAND_MAX);
    }
    std::complex<FLT> numerical;
    theia::cubature::cauchy::eval<FLT,DIM>(rule,f,b,numerical);
    auto exact = f(b);
    num += std::abs(numerical - exact);
    den += std::abs(exact);
  }

  std::cout << std::boolalpha << (num/den < 1.e-6) << std::endl;
  
  return 0;
}
