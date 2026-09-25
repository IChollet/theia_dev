#ifndef THEIA_QUADRATURE_GAUSS_HERMITE_HPP
#define THEIA_QUADRATURE_GAUSS_HERMITE_HPP

#include "../../matrices.hpp"
#include "../cubature_struct.hpp"

namespace theia{
  namespace cubature{
    namespace gauss_hermite{

      template<typename FLT> using rule_type = rule<FLT,1,FLT,FLT>;

      /*
	Orthonormal Hermite polynomials (weight e^{-t^2}):
	  h_0 = pi^{-1/4},  h_{k+1} = sqrt(2/(k+1)) t h_k - sqrt(k/(k+1)) h_{k-1},
	  h_n' = sqrt(2n) h_{n-1}
	Returns h_n(t), h_{n-1}(t) and sum_{k<n} h_k(t)^2 (Christoffel function).
	Computed in long double: the values grow like e^{t^2/2} in the tails.
      */
      inline void orthonormal_hermite(long double t, int n,
				      long double& hn, long double& hn1, long double& sum2){
	long double h0 = std::pow(3.141592653589793238462643383279502884L,-0.25L), h1 = 0.L;
	sum2 = 0.L;
	for(int k = 0; k < n; k++){
	  sum2 += h0*h0;
	  long double h2 = std::sqrt(2.L/(k+1)) * t * h0 - std::sqrt((long double)k/(k+1)) * h1;
	  h1 = h0; h0 = h2;
	}
	hn = h0; hn1 = h1;
      }

      // Newton iterations on h_n from an initial guess t (a few steps at most)
      inline long double refine_node(long double t, int n){
	for(int it = 0; it < 3; it++){
	  long double hn, hn1, sum2;
	  orthonormal_hermite(t, n, hn, hn1, sum2);
	  long double dt = hn / (std::sqrt(2.L*n) * hn1);
	  t -= dt;
	  if(std::abs(dt) <= 1.e-18L * std::max(1.L, std::abs(t))){break;}
	}
	return t;
      }

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
	// Golub-Welsch nodes t_i (weight e^{-t^2}) refined by Newton; weights from the
	// Christoffel function w_i = 1 / sum_{k<n} h_k(t_i)^2, accurate in relative
	// precision (even for tiny tail weights). Then x = sqrt(2) t (weight e^{-x^2/2})
	for(int i = 0; i < order; ++i){
	  long double t = refine_node(rule1D.x[i][0], order);
	  long double hn, hn1, sum2;
	  orthonormal_hermite(t, order, hn, hn1, sum2);
	  rule1D.x[i][0] = FLT(std::sqrt(2.L) * t);
	  rule1D.w[i]    = FLT(std::sqrt(2.L) / sum2);
	}
	(void)sqrtPi;
	return rule1D;
      }
      
    } // GAUSS_HERMITE
  } // CUBATURE
} // THEIA

#endif
