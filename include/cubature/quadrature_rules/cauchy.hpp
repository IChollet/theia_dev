#ifndef THEIA_CUBATURE_CAUCHY_HPP
#define THEIA_CUBATURE_CAUCHY_HPP
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>
#include "../../matrices.hpp"
#include "../cubature_struct.hpp"

namespace theia{
  namespace cubature{
    namespace cauchy{

      template<typename FLT> using rule_type = rule<FLT,1,
						    std::complex<FLT>,
						    std::complex<FLT>>;

      template<class FLT>
      rule_type<FLT> get(int                      number_of_quadrature_points,
			 const std::complex<FLT>& center,
			 const std::complex<FLT>& radius){
	
	// Checkings
	static_assert(std::is_same_v<FLT,float> || std::is_same_v<FLT,double>,
                      "FLT must be float or double");
        if(number_of_quadrature_points < 1){
          std::cout << "order < 1 in cauchy cubature computation" << std::endl;exit(1);}

	// Result declaration
	rule_type<FLT> result;
        result.N = number_of_quadrature_points;
        result.x.resize(number_of_quadrature_points);
	result.w.resize(number_of_quadrature_points);

	// Computations
	FLT dtheta = 2.*M_PI / FLT(number_of_quadrature_points);
	FLT w      = 1.      / FLT(number_of_quadrature_points);
	for(int i = 0; i < number_of_quadrature_points; i++){
	  FLT theta = FLT(i) * dtheta;
	  std::complex<FLT> e             = radius*std::exp(std::complex<FLT>(0.,theta));
	  std::complex<FLT> z_theta       = center+e;
	  std::complex<FLT> z_theta_prime = e;
	  result.w[i]    =  w * z_theta_prime;
	  result.x[i][0] =  z_theta;
	}

	// Output
	return result;
      }

      /*
	Cauchy rule for a given point: the factor 1/(z_i - point) is included
	in the weights, so that the generic eval(rule, f, res) gives
	   (1/2 i pi) \oint f(z)/(z - point) dz   ( = f(point) if point is inside the circle,
	                                           0        if it is outside )
	Being a standard rule, it can be tensorised with any other rule
	(e.g. one Cauchy rule per dimension, each with its own point).
      */
      template<class FLT>
      rule_type<FLT> get(int                      number_of_quadrature_points,
			 const std::complex<FLT>& center,
			 const std::complex<FLT>& radius,
			 const std::complex<FLT>& point){
	rule_type<FLT> result = get<FLT>(number_of_quadrature_points, center, radius);
	for(int i = 0; i < result.N; i++){
	  std::complex<FLT> d = result.x[i][0] - point;
	  if(d == std::complex<FLT>(0.)){
	    std::cout << "point lies on a node of the cauchy rule" << std::endl; exit(1);}
	  result.w[i] /= d;
	}
	return result;
      }


      /*
	TEMPLATE ARGS:
	- OUT_t denotes the return type
	- F     denotes the integrand functor types
      */ 
      template<typename FLT, int DIM>
      static void eval(const rule<FLT,DIM,std::complex<FLT>,std::complex<FLT>>& r,
		       std::function<std::complex<FLT>
		       (const std::array<std::complex<FLT>,DIM>&)> f,
		       const std::array<std::complex<FLT>,DIM>& point,
		       std::complex<FLT>& res){
	res = std::complex<FLT>(0.);
	for(int i = 0; i < r.N; i++){
	  std::complex<FLT> div = 1.;
	  for(int k = 0; k < DIM; k++){ div /= (r.x[i][k] - point[k]);}
	  res += r.w[i] * f(r.x[i]) * div;
	}
      }
      
    } // CAUCHY
  } // CUBATURE
} // THEIA

#endif
