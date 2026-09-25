#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/algorithms/beylkin_monzon.hpp"  

#define FLT double

int main(int argc, char* argv[]){

  const FLT Rm = 1.;
  const FLT Rp = 10.;
  
  auto res = theia::beylkin_monzon::approximation<1,FLT,std::complex<FLT>>
    ([](FLT R){
      return std::exp(0.1*std::complex<FLT>(0.,R))/(R);
    }, Rm, Rp, 1e-8);

  /*
  std::cout << res.relative_error << "\t" << res.M << std::endl;
  for(int i = 0; i < res.M; i++){
    std::cout << res.coefficients[i] << "\t" << res.modes[i] << std::endl;
  }
  */
  std::cout << std::boolalpha << (res.relative_error < 1.e-8) << std::endl;
  
  return 0;
}
