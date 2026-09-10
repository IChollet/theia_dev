#ifndef THEIA_QUADRATURE_GAUSS_HERMITE_HPP
#define THEIA_QUADRATURE_GAUSS_HERMITE_HPP

#include "../../matrices.hpp"
#include "../cubature_struct.hpp"

namespace theia{
  namespace cubature{
    namespace gauss_hermite{

      template<typename FLT> using rule_type = rule<FLT,1,FLT,FLT>;
      
      template<class FLT> rule_type<FLT> get(int order){

	// Check template value and args
	static_assert(std::is_same<FLT,float >::value ||
		      std::is_same<FLT,double>::value,
		      "Gauss-Hermite rule only defined for float / double");
	if(order < 1){
	  std::cout << "order < 1 in gauss-hermite computation" << std::endl;
	  exit(1);
	}

	// Initialise rule and set constants
	rule<FLT,1,FLT,FLT> rule1D;
	rule1D.x.resize(order);
	rule1D.w.resize(order);
	rule1D.N     = order;
	constexpr long double pi = 3.141592653589793238462643383279502884L;
	const FLT sqrtPi = std::sqrt(static_cast<FLT>(pi));
    
	// Compute the rule
	const int N = static_cast<std::size_t>(order);
	std::vector<FLT> J(N*N, FLT(0.));
	for(int k = 0; k < order-1; ++k){
	  const FLT b = std::sqrt(FLT(k+1)/FLT(2.));
	  J[k + (k+1)*order] = b;
	  J[(k+1) + k*order] = b;
	}
	theia::syev<FLT>(order, J.data(), (FLT*)rule1D.x.data());
	for(int i = 0; i < order; ++i){
	  rule1D.x[i][0] *= std::sqrt(FLT(2));
	  const FLT v0 = J[i*order];
	  rule1D.w[i] = sqrtPi * v0 * v0 * std::sqrt(FLT(2));
	}
	return rule1D;
      }
      
    } // GAUSS_HERMITE
  } // CUBATURE
} // THEIA

#endif
