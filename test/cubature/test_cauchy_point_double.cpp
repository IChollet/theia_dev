#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/cubature/cubature.hpp"

#define FLT double
#define CPLX std::complex<FLT>

using namespace theia::cubature;

/*
  Point-dependent Cauchy rule (1/(z-p) included in the weights):
   - 1D: same result as cauchy::eval, and f(p) inside / 0 outside the circle
   - 2D: tensor of two rules with their own point -> f(p1,p2)
*/
int main(){
  bool ok = true;
  auto f1 = [](const std::array<CPLX,1>& z){return std::exp(z[0])/(z[0]+CPLX(3.,1.));};
  CPLX c(8.,0.5), p(8.2,0.3);
  FLT  r = 0.5;

  // 1D
  auto rp = cauchy::get<FLT>(80, c, CPLX(r), p);
  auto r0 = cauchy::get<FLT>(80, c, CPLX(r));
  CPLX ip, i0;
  eval(rp, f1, ip);
  cauchy::eval<FLT,1>(r0, f1, {p}, i0);
  ok &= (std::abs(ip-i0)     /std::abs(i0)      < 1.e-13);
  ok &= (std::abs(ip-f1({p}))/std::abs(f1({p})) < 1.e-12);
  auto rout = cauchy::get<FLT>(80, c, CPLX(r), CPLX(10.,0.));   // point outside the circle
  CPLX iout;
  eval(rout, f1, iout);
  ok &= (std::abs(iout)/std::abs(f1({p})) < 1.e-12);

  // 2D: different circles and points per dimension
  auto f2 = [](const std::array<CPLX,2>& z){return std::exp(z[0]*z[1])/(z[0]-z[1]+CPLX(10.));};
  CPLX p1(0.1,0.2), p2(2.9,-0.1);
  auto ra = cauchy::get<FLT>(40, CPLX(0.,0.), CPLX(0.6), p1);
  auto rb = cauchy::get<FLT>(40, CPLX(3.,0.), CPLX(0.4), p2);
  auto r2 = tensor_rule(ra,rb);
  CPLX i2;
  eval(r2, f2, i2);
  ok &= (std::abs(i2-f2({p1,p2}))/std::abs(f2({p1,p2})) < 1.e-12);

  std::cout << std::boolalpha << ok << std::endl;
  return 0;
}
