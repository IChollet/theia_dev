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
	constexpr char JOBZ  = 'V';
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
	for (int i = 0; i < order; i++){
	  const FLT v0 = eigvecs[i*order];
	  result.w[i] = v0 * v0;
	}
	return result;
      }
      
    } // GAUSS_LAGUERRE
  } // CUBATURE
} // THEIA
#endif
