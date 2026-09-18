#include <iostream>
#include "../../include/cubature/cubature.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <vector>

#define FLT double
#define DIM 3

int main(){
  constexpr int    dim     = DIM;
  constexpr int    Ntest   = 10000;
  constexpr int    order   = 2;
  constexpr FLT    scaling = 1.;
  std::vector<FLT> b(dim);

  // Fonction f(x) = x^2 + y^2 + z^2 ...
  auto integrand = [&](const std::array<FLT,DIM>& x){
    FLT dot = 0.;
    for(int i = 0; i < dim; ++i)
      dot += x[i] * x[i];
    return dot;
  };
  
  auto rule  = theia::cubature::gauss_legendre::get<FLT>(order);
  auto rule2 = theia::cubature::tensor_rule(rule,rule);
  auto rule3 = theia::cubature::tensor_rule(rule2,rule);  

  FLT num = 0.;
  FLT den = 0.;
  for(int jj = 0; jj < Ntest; ++jj){
    for(int i = 0; i < dim; ++i){
      b[i] = scaling * (std::rand() / FLT(RAND_MAX) > 0.5 ? 1. : -1.)
	* std::rand() / FLT(RAND_MAX);
    }
    FLT numerical;
    eval(rule3,integrand,numerical); // En dimension 3
    FLT norm2 = 0.0; for(FLT v : b) norm2 += v * v;    
    const FLT exact = std::pow(2.0, dim);
    num += std::abs(numerical - exact);
    den += std::abs(exact);
  }

  std::cout << std::boolalpha << (num/den < 1.e-10) << std::endl;

  return 0;

}
