#ifndef THEIA_QUADRATURE_GAUSS_LAGUERRE_HPP
#define THEIA_QUADRATURE_GAUSS_LAGUERRE_HPP
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "../cubature_struct.hpp"
#include "../../matrices.hpp"

namespace theia{

  namespace cubature{

    namespace gauss_laguerre{

      template<typename FLT> using rule_type = rule<FLT,1,FLT,FLT>;

      /*
	Laguerre polynomials (orthonormal for the weight e^{-x}):
	  L_0 = 1,  (k+1) L_{k+1} = (2k+1-x) L_k - k L_{k-1},   x L_n' = n (L_n - L_{n-1})
	Returns L_n(x), L_{n-1}(x) and sum_{k<n} L_k(x)^2 (Christoffel function).
	Computed in long double: the values grow like e^{x/2} in the tails.
      */
      inline void laguerre(long double x, int n,
			   long double& ln, long double& ln1, long double& sum2){
	long double l0 = 1.L, l1 = 0.L;
	sum2 = 0.L;
	for(int k = 0; k < n; k++){
	  sum2 += l0*l0;
	  long double l2 = ((2.L*k+1.L-x) * l0 - (long double)k * l1) / (k+1);
	  l1 = l0; l0 = l2;
	}
	ln = l0; ln1 = l1;
      }

      // Newton iterations on L_n from an initial guess x (a few steps at most)
      inline long double refine_node(long double x, int n){
	for(int it = 0; it < 3; it++){
	  long double ln, ln1, sum2;
	  laguerre(x, n, ln, ln1, sum2);
	  long double dx = ln * x / (n * (ln - ln1));
	  x -= dx;
	  if(std::abs(dx) <= 1.e-18L * std::max(1.L, std::abs(x))){break;}
	}
	return x;
      }

      template<class FLT> rule_type<FLT> get(int order){
	// Check template value and args
	static_assert(std::is_same_v<FLT,float> || std::is_same_v<FLT,double>,
		      "FLT must be float or double");
	if(order < 1){
	  std::cout << "order < 1 in gauss-laguerre computation" << std::endl;
	  exit(1);
	}
	
	// Compute the rule (using Jacobi matrix)
	std::vector<FLT> d(order);
	std::vector<FLT> e(std::max(0, order - 1));
	for (int i = 0; i < order; i++) d[i] = FLT(2 * i + 1);
	for (int i = 0; i < order - 1; i++) e[i] = FLT(i + 1);
	std::vector<FLT> eigvals(order);
	std::vector<FLT> eigvecs(order * order);
	constexpr char JOBZ  = 'N'; // eigenvalues only: weights are computed below
	constexpr char RANGE = 'A';
	int m      = 0;
	int ldz    = order;
	int info   = 0;
	FLT vl = FLT(0);
	FLT vu = FLT(0);
	int il = 0;
	int iu = 0;
	FLT abstol = FLT(0);
	std::vector<int> isuppz(2 * std::max(1, order));
	int lwork  = -1;
	int liwork = -1;
	if constexpr (std::is_same_v<FLT,double>){
	  double work_query;
	  int iwork_query;
	  dstevr_(&JOBZ, &RANGE,&order,d.data(),e.data(),&vl, &vu,&il, &iu,&abstol,&m,eigvals.data(),eigvecs.data(),&ldz,isuppz.data(),&work_query,&lwork,&iwork_query,&liwork,&info);
	  lwork  = static_cast<int>(work_query);
	  liwork = iwork_query;
	  std::vector<double> work(lwork);
	  std::vector<int>    iwork(liwork);
	  dstevr_(&JOBZ, &RANGE,&order,d.data(),e.data(),&vl, &vu,&il, &iu,&abstol,&m,eigvals.data(),eigvecs.data(),&ldz,isuppz.data(),work.data(),&lwork,iwork.data(),&liwork,&info);
	}else{
	  float work_query;
	  int iwork_query;
	  sstevr_(&JOBZ, &RANGE,&order,d.data(),e.data(),&vl, &vu,&il, &iu,&abstol,&m,eigvals.data(),eigvecs.data(),&ldz,isuppz.data(),&work_query,&lwork,&iwork_query,&liwork,&info);
	  lwork  = static_cast<int>(work_query);
	  liwork = iwork_query;
	  std::vector<float> work(lwork);
	  std::vector<int>    iwork(liwork);
	  sstevr_(&JOBZ, &RANGE,&order,d.data(),e.data(),&vl, &vu,&il, &iu,&abstol,&m,eigvals.data(),eigvecs.data(),&ldz,isuppz.data(),work.data(),&lwork,iwork.data(),&liwork,&info);
	}
	if (info != 0){std::cout << "STEVR failed in gauss-laguerre computation" << std::endl; exit(1);}

	// Set the result
        rule_type<FLT> result;
	result.N = order;
	result.x.resize(order);
	for(int i = 0; i < order; i++){result.x[i][0] = eigvals[i];}
	result.w.resize(order);
	// Golub-Welsch nodes refined by Newton; weights from the Christoffel function
	// w_i = 1 / sum_{k<n} L_k(x_i)^2, accurate in relative precision (even for
	// tiny tail weights)
	for (int i = 0; i < order; i++){
	  long double x = refine_node(result.x[i][0], order);
	  long double ln, ln1, sum2;
	  laguerre(x, order, ln, ln1, sum2);
	  result.x[i][0] = FLT(x);
	  result.w[i]    = FLT(1.L / sum2);
	}
	return result;
      }
      
    } // GAUSS_LAGUERRE
  } // CUBATURE
} // THEIA
#endif
