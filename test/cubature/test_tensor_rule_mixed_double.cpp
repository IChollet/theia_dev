#include <iomanip>
#include <cmath>
#include <iostream>
#include <complex>
#include "../../include/cubature/cubature.hpp"

#define FLT double
#define CPLX std::complex<FLT>

using namespace theia::cubature;

/*
  Generic tensor_rule:
   - sizes are correct whatever the order of the arguments (N1*N2)
   - real x real rules of different sizes integrate polynomials exactly
   - real x complex (and float x complex<double>) rules: promoted types and values
*/
int main(){
  bool ok = true;

  // Sizes and exactness, real x real, both orders
  auto r  = gauss_legendre::get<FLT>(3);        // exact up to degree 5
  auto r2 = tensor_rule(r,r);
  auto a  = tensor_rule(r2,r);
  auto b  = tensor_rule(r,r2);
  auto r4 = tensor_rule<FLT,1,FLT,FLT,4>(r);
  ok &= (r2.N == 9 && a.N == 27 && b.N == 27 && r4.N == 81);
  ok &= (int(a.x.size()) == a.N && int(a.w.size()) == a.N && int(r4.x.size()) == r4.N);
  // int_[-1,1]^3 (1+x)^2 y^4 z^2 = 8/3 * 2/5 * 2/3
  auto poly = [](const std::array<FLT,3>& x){return (1.+x[0])*(1.+x[0]) * std::pow(x[1],4) * x[2]*x[2];};
  FLT ia, ib;
  eval(a,poly,ia);
  eval(b,poly,ib);
  const FLT exact_poly = 8./3. * 2./5. * 2./3.;
  ok &= (std::abs(ia-exact_poly) < 1.e-14 && std::abs(ib-exact_poly) < 1.e-14);

  // Real (Gauss-Legendre) x complex (Cauchy at a point), both orders
  //   int_{-1}^{1} e^x dx * g(p)
  CPLX p(0.2,-0.1);
  auto g  = [](CPLX z){return std::exp(z)*std::cos(z);};
  auto gl = gauss_legendre::get<FLT>(12);
  auto ca = cauchy::get<FLT>(32, CPLX(0.), CPLX(1.), p);
  auto t1 = tensor_rule(gl,ca);
  auto t2 = tensor_rule(ca,gl);
  static_assert(std::is_same_v<decltype(t1), rule<FLT,2,CPLX,CPLX>>, "real x complex -> complex");
  static_assert(std::is_same_v<decltype(t2), rule<FLT,2,CPLX,CPLX>>, "complex x real -> complex");
  static_assert(std::is_same_v<decltype(r2), rule<FLT,2,FLT ,FLT >>, "real x real unchanged");
  const CPLX exact = (std::exp(1.)-std::exp(-1.)) * g(p);
  CPLX i1, i2;
  eval(t1, [&](const std::array<CPLX,2>& x){return std::exp(x[0]) * g(x[1]);}, i1);
  eval(t2, [&](const std::array<CPLX,2>& x){return std::exp(x[1]) * g(x[0]);}, i2);
  ok &= (std::abs(i1-exact)/std::abs(exact) < 1.e-13);
  ok &= (std::abs(i2-exact)/std::abs(exact) < 1.e-13);

  // Mixed precisions: float x complex<double> -> complex<double>
  auto glf = gauss_legendre::get<float>(6);
  auto t3  = tensor_rule(glf,ca);
  static_assert(std::is_same_v<decltype(t3), rule<FLT,2,CPLX,CPLX>>, "float x complex<double> -> complex<double>");
  CPLX i3;
  eval(t3, [&](const std::array<CPLX,2>& x){return std::exp(x[0]) * g(x[1]);}, i3);
  ok &= (std::abs(i3-exact)/std::abs(exact) < 1.e-6);

  std::cout << std::boolalpha << ok << std::endl;
  return 0;
}
