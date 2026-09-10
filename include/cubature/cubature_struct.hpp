#ifndef THEIA_CUBATURE_STRUCT_HPP
#define THEIA_CUBATURE_STRUCT_HPP
#include <iostream>
#include <vector>
#include <complex>
#include <functional>
#include <cmath>
#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace theia{
  namespace cubature{

    /*
      TEMPLATE ARGS:
      - FLT is the artihmetic precision used in the computation of the rule
      - DIM is the dimension of nodes
      - NODE_t denotes the node value type (real, complex, etc...)
      - WGHT_t denotes the weight type
    */
    template<typename FLT, int DIM, typename NODE_t, typename WGHT_t>
    struct rule{
      int                                 N; // Total number of points
      std::vector<std::array<NODE_t,DIM>> x; // List of nodes of the quadrature rule
      std::vector<WGHT_t>                 w; // List of weights of the quadrature
    }; // RULE

    template<typename FLT, int DIM, typename NODE_t, typename WGHT_t>
    void print(rule<FLT,DIM,NODE_t,WGHT_t>& r){
      std::cout << "Cubature rule using " << r.N << " nodes:" << std::endl;
      for(int i = 0; i < r.N; i++){
	std::cout << "Nodes " << i << ": ";
	for(int k = 0; k < DIM; k++){
	  std::cout << r.x[i][k] << "\t";
	}
	std::cout << "Weight: " << r.w[i] << std::endl;
      }
    }

    /*
      TEMPLATE ARGS:
         - OUT_t denotes the return type
	 - F     denotes the integrand functor types
    */ 
    template<typename FLT, int DIM, typename NODE_t, typename WGHT_t, typename OUT_t, class F>
    static void eval(const rule<FLT,DIM,NODE_t,WGHT_t>& r, F&& f, OUT_t& res){
      res = OUT_t(0.);
      for(int i = 0; i < r.N; i++){
	res += r.w[i] * f(r.x[i]);
	std::cout << "\t" << r.w[i] << "\t" << f(r.x[i]) << "\t" << res << std::endl;
      }
    }

    template<typename FLT, int DIM_1, int DIM_2, typename NODE_t, typename WGHT_t>
    static rule<FLT,DIM_1+DIM_2,NODE_t,WGHT_t> tensor_rule(const rule<FLT,DIM_1,NODE_t,WGHT_t>& r1,
						    const rule<FLT,DIM_2,NODE_t,WGHT_t>& r2){
      rule<FLT,DIM_1+DIM_2,NODE_t,WGHT_t> tensor_rule;
      tensor_rule.N = r1.N*r1.N;
      tensor_rule.x.resize  (tensor_rule.N);
      tensor_rule.w.resize(tensor_rule.N);
      for(int i = 0; i < r1.N; i++){
	for(int j = 0; j < r2.N; j++){
	  tensor_rule.w[i*r2.N+j] = r1.w[i]*r2.w[j];
	  for(int k = 0; k < DIM_1; k++){
	    tensor_rule.x[i*r2.N+j][      k] = r1.x[i][k];
	  }
	  for(int k = 0; k < DIM_2; k++){
	    tensor_rule.x[i*r2.N+j][DIM_1+k] = r2.x[j][k];
	  }
	}
      }
      return tensor_rule;
    }

    // Generation for arbitrary dimension
    template<typename FLT, int DIM, typename NODE_t, typename WGHT_t, int TENSOR>
    auto tensor_rule(const rule<FLT,DIM,NODE_t,WGHT_t>& r){
      static_assert(TENSOR > 0, "TENSOR must be > 0");
      if constexpr (TENSOR == 1){ auto tr = r; return r;}
      else{
	auto tr = tensor_rule(tensor_rule<FLT,DIM,NODE_t,WGHT_t,TENSOR-1>(r),r);
	return tr;}
    }
   
    
  }// CUBATURE  
} // THEIA

#endif // THEIA_CUBATURE_STRUCT_HPP
