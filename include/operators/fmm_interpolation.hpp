#ifndef THEIA_OPERATORS_FMM_INTERPOLATION_HPP
#define THEIA_OPERATORS_FMM_INTERPOLATION_HPP

#include <iostream>
#include <cmath>
#include <array>
#include "../general_intrp.hpp"

namespace theia{

  template<size_t DIM, typename FLT, typename T, int ITYPE>
  struct op_interpolation{}; // OP_INTERPOLATION

  template<size_t DIM, typename FLT, typename T, int ITYPE>
  inline void P2M(FLT* mins, FLT* maxs,
		  int N, std::array<FLT,DIM>* py,
		  int L, T*& A, const op_interpolation<DIM,FLT,T,ITYPE>& info){ 
    get_polynomials<DIM,FLT,T,ITYPE>(L,A,mins,maxs,py,N);
  }

  template<size_t DIM, typename FLT, typename T, class KRNL, int ITYPE>
  inline void M2L(FLT* minsX, FLT* maxsX, int Lx,
		  FLT* minsY, FLT* maxsY, int Ly,
		  KRNL& K, T*& A, const op_interpolation<DIM,FLT,T,ITYPE>& info){
    int Lxd  = myintpow(Lx,DIM);
    int Lyd  = myintpow(Lx,DIM);
    if(A == nullptr){
      A      = new T  [Lxd*Lyd];
    }
    std::array<FLT,DIM> *px = new std::array<FLT,DIM>[Lxd];
    std::array<FLT,DIM> *py = new std::array<FLT,DIM>[Lyd];
    get_multivariate_interp_nodes<DIM,FLT,ITYPE>(Lx,minsX,maxsX,px);
    get_multivariate_interp_nodes<DIM,FLT,ITYPE>(Ly,minsY,maxsY,py);
    K(px,Lxd,py,Lyd,A);
  }

  template<size_t DIM, typename FLT, typename T, int ITYPE>
  inline void M2M(int * left_L, FLT*  left_mins, FLT*  left_maxs,
		  int *right_L, FLT* right_mins, FLT* right_maxs,
		  Kron<DIM,T>& A, const op_interpolation<DIM,FLT,T,ITYPE>& info){
    T **S  = new T*[DIM];
    for(int d = 0; d < DIM; d++){
      S[d] = nullptr;
      std::array<FLT,1> *px = new std::array<FLT,1>[ left_L[d]];
      std::array<FLT,1> *py = new std::array<FLT,1>[right_L[d]];
      get_multivariate_interp_nodes<1,FLT,ITYPE>
	(right_L[d], right_mins+d, right_maxs+d, py);
      get_polynomials<1,FLT,T,ITYPE>
	(left_L[d], S[d], left_mins+d, left_maxs+d, py, right_L[d]);
    }
    A.set(S,left_L,right_L);
  }

} // THEIA

#endif
