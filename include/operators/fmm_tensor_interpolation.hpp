#ifndef THEIA_OPERATORS_FMM_TENSOR_INTERPOLATION_HPP
#define THEIA_OPERATORS_FMM_TENSOR_INTERPOLATION_HPP

#include <iostream>
#include <cmath>
#include <array>
#include "../general_intrp.hpp"
#include "../tensor_p2m.hpp"
#include "./fmm_interpolation.hpp"

namespace theia{

  /*
    Same interpolation as op_interpolation, but P2M (and L2P = P2M^T) are kept
    in tensor form (TensorP2M): only the 1D Lagrange polynomials of each
    direction are stored, and products are computed particle by particle.
  */
  template<size_t DIM, typename FLT, typename T, int ITYPE>
  struct op_tensor_interpolation{}; // OP_TENSOR_INTERPOLATION

  template<size_t DIM, typename FLT, typename T, int ITYPE>
  inline void P2M(FLT* mins, FLT* maxs,
		  int N, std::array<FLT,DIM>* py,
		  const int* L, TensorP2M<DIM,T>& A, const op_tensor_interpolation<DIM,FLT,T,ITYPE>& info){
    T*  S [DIM];
    int Ls[DIM];
    std::array<FLT,1> *pk = new std::array<FLT,1>[N];
    for(size_t d = 0; d < DIM; d++){
      Ls[d] = L[d];
      S [d] = nullptr;
      for(int j = 0; j < N; j++){pk[j][0] = py[j][d];}
      get_polynomials<1,FLT,T,ITYPE>(L[d], S[d], mins+d, maxs+d, pk, N);
    }
    A.set(S,Ls,N);
    for(size_t d = 0; d < DIM; d++){delete [] S[d];}
    delete [] pk;
  }

  // M2L and M2M do not depend on the P2M representation
  template<size_t DIM, typename FLT, typename T, class KRNL, int ITYPE>
  inline void M2L(FLT* minsX, FLT* maxsX, const int* Lx,
		  FLT* minsY, FLT* maxsY, const int* Ly,
		  KRNL& K, T*& A, const op_tensor_interpolation<DIM,FLT,T,ITYPE>& info){
    M2L(minsX,maxsX,Lx,minsY,maxsY,Ly,K,A,op_interpolation<DIM,FLT,T,ITYPE>());
  }

  template<size_t DIM, typename FLT, typename T, int ITYPE>
  inline void M2M(int * left_L, FLT*  left_mins, FLT*  left_maxs,
		  int *right_L, FLT* right_mins, FLT* right_maxs,
		  Kron<DIM,T>& A, const op_tensor_interpolation<DIM,FLT,T,ITYPE>& info){
    M2M(left_L,left_mins,left_maxs,right_L,right_mins,right_maxs,A,
	op_interpolation<DIM,FLT,T,ITYPE>());
  }

} // THEIA

#endif
