#include <iostream>
#include "../include/cubature/cubature.hpp"
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
  constexpr int    order   = 8;
  constexpr FLT    scaling = 1.;
  std::vector<FLT> b(dim);

  // Fonction f(x) = e^{<b,x>}
  auto integrand = [&](const std::array<FLT,DIM>& x){
    FLT dot = 0.;
    for(int i = 0; i < dim; ++i)
      dot += b[i] * x[i];
    return std::exp(dot);
  };
  
  auto rule  = theia::cubature::gauss_hermite::get<FLT>(order);
  print(rule);
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
    const FLT exact = std::pow(2.0 * M_PI, dim / 2.0) * std::exp(0.5 * norm2);
    num += std::abs(numerical - exact);
    den += std::abs(exact);
  }
  std::cout << std::setprecision(16);
  std::cout << "Nombre de points : " << std::pow(order, dim) << "\n";
  std::cout << "Rel error : " << num / den << "\n";
}
