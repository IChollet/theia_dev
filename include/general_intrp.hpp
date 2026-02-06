//===================================================================
//
// Authors: Igor Chollet
//
//  This file is part of theia.
//
//  theia is free software: you can redistribute it and/or modify
//  it under the terms of the GNU Lesser General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//
//  theia is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU Lesser General Public License for more details.
//  (see LICENSE.txt)
//
//  You should have received a copy of the GNU Lesser General Public License
//  along with theia.  If not, see <http://www.gnu.org/licenses/>
//
//====================================================================
#ifndef THEIA_GENERAL_INTRP_HPP
#define THEIA_GENERAL_INTRP_HPP
#include <iostream>
#include <cmath>
#include <array>
#include "./polynomials.hpp"

namespace theia{
  
  template<int DIM, typename FLT, int ITYPE>
  inline void get_multivariate_interp_nodes(int L,
					    FLT* mins, FLT* maxs,
					    std::array<FLT,DIM>* z){
    int Ld = myintpow(L,DIM);
    for(int i = 0; i < Ld; i++){
      int itmp = i;
      for(int k = 0; k < DIM; k++){
	int ki        = (itmp % L);
	itmp          = (itmp / L);
	z[i][DIM-1-k] = get_node<ITYPE>(ki,L,mins[DIM-1-k],maxs[DIM-1-k]);
      }
    }
  }
  
  template<int DIM, typename FLT, typename T, int ITYPE>
  inline void get_polynomials(int L, T*& S,
			      FLT* mins, FLT* maxs,
			      std::array<FLT,DIM>* prts, int N){
    int Ld  = myintpow(L,DIM);
    S       = new T   [Ld*N];
    FLT **V = new FLT*[DIM];
    for(int k = 0; k < DIM; k++){
      V[k] = new FLT[L*N];
      FLT* Vk = V[k];
      FLT ctr = mins[k]+maxs[k];
      ctr    *= 0.5;
      FLT rad = std::abs(ctr-maxs[k]);
      for(int j = 0; j < N; j++){
	FLT y     = (prts[j][k]-ctr)/rad;
        for(int i = 0; i < L; i++){
	  Vk[j*L+i] = C1D<ITYPE>(y,i,L);
	}
      }
    }
    for(int i = 0; i < Ld*N; i++){S[i] = FLT(1.);}
    for(int j = 0; j < N; j++){
      for(int i = 0; i < Ld; i++){
	int ti = i;
	for(int k = 0; k < DIM; k++){
	  int ri = (ti % L);
	  ti     = (ti / L);
	  S[j*Ld + i] *= V[DIM-k-1][j*L+ri];
	}
      }
    }
  }

  template<int DIM, typename FLT, int LEFT_TYPE, int RIGHT_TYPE>
  inline void get_reinterpolation_matrices(int * left_L, FLT*  left_mins, FLT*  left_maxs,
					   int *right_L, FLT* right_mins, FLT* right_maxs,
					   Kron<DIM,FLT>& A){
    FLT **S  = new FLT*[DIM];
    for(int d = 0; d < DIM; d++){
      S [d]     = new FLT[left_L[d]*right_L[d]];
      FLT ctrld = ( left_mins[d] +  left_maxs[d])*0.5;
      FLT ctrrd = (right_mins[d] + right_maxs[d])*0.5;
      FLT radld = std::abs( left_mins[d] -  left_maxs[d])*0.5;
      FLT radrd = std::abs(right_mins[d] - right_maxs[d])*0.5;
      for(int j = 0; j < right_L[d]; j++){
	FLT y = (ctrrd + radrd * get_node<RIGHT_TYPE>(j,right_L[d]) - ctrld) / radld;
        for(int i = 0; i < left_L[d]; i++){
	  S[d][j*left_L[d]+i] = C1D<LEFT_TYPE>(y,i,left_L[d]);
	}
      }
    }
    A(S,left_L,right_L);
  }
  
  // Templates :
  //     T     : Kernel type
  //     FLT   : Floating point type (i.e. Lagrange pol. type)
  //     DIM   : Ambiant dimension
  //     KRNL  : Kernel class type
  //     ITYPE : Interpolation type (0 for Chebyshev / 1 for equispaced)
  template<typename T, typename FLT, int DIM, class KRNL, int ITYPE>  class lits{
  private:
    
    T*      Sl;                  // Left    polynomials
    T*      Sr;                  // Right   polynomials
    T*      A;                   // Central symbolic matrix
    int     L;                   // Interpolation order
    FLT*    minsl;               // Left    lower interval bounds
    FLT*    maxsl;               // Left    maximal interval bounds
    int     Nl;                  // Left    number of particles
    std::array<FLT,DIM>* prtsl;  // Left    particles
    FLT*    minsr;               // Right   [...]
    FLT*    maxsr;               // Right   [...]
    int     Nr;                  // Right   [...]
    std::array<FLT,DIM>* prtsr;  // Right   [...]
    int    r;                    // Left and right ranks
    lrmat<T> UV;                 // Low-rank factorization of A
    T*     SlU;                  // Final left term 
    T*     VSr;                  // Final right term
    KRNL*  K;                    // Kernel reference
    
  public :
    lits(){}
    lits(FLT* minsl_, FLT* maxsl_, std::array<FLT,DIM>* prtsl_, int Nl_,
	     FLT* minsr_, FLT* maxsr_, std::array<FLT,DIM>* prtsr_, int Nr_,
	     int L_, KRNL* K_){
      L = L_; prtsl = prtsl_; Nl = Nl_; K = K_;
      minsl = new FLT[DIM];
      maxsl = new FLT[DIM];
      for(int k = 0; k < DIM; k++){
	minsl[k] = minsl_[k];
	maxsl[k] = maxsl_[k];
      }
      prtsr = prtsr_; Nr = Nr_;
      minsr = new FLT[DIM];
      maxsr = new FLT[DIM];
      for(int k = 0; k < DIM; k++){
	minsr[k] = minsr_[k];
	maxsr[k] = maxsr_[k];
      }
    }

    void get_source_nodes(std::array<FLT,DIM>*& py, int& number_of_nodes){
      number_of_nodes = myintpow(L,DIM);
      py = new std::array<FLT,DIM>[number_of_nodes];
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsr,maxsr,py);}
    
    void get_target_nodes(std::array<FLT,DIM>*& px, int& number_of_nodes){
      number_of_nodes = myintpow(L,DIM);
      px = new std::array<FLT,DIM>[number_of_nodes];
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsl,maxsl,px);}
    
    void get_UV(double epsilon){
      int Ld  = myintpow(L,DIM);
      A       = new T  [Ld*Ld];
      std::array<FLT,DIM> *px = new std::array<FLT,DIM>[Ld];
      std::array<FLT,DIM> *py = new std::array<FLT,DIM>[Ld];
      r = myintpow(L,DIM);
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsr,maxsr,py);
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsl,maxsl,px);
      get_symbolic_matrix<DIM,FLT,T,KRNL>(px,py,r,r,A,K);
      get_polynomials<DIM,FLT,T,ITYPE>(L,Sr,minsr,maxsr,prtsr,Nr);
      get_polynomials<DIM,FLT,T,ITYPE>(L,Sl,minsl,maxsl,prtsl,Nl);
      gesvd<T>(A,r,UV,epsilon);
      //paca<T>(A,r,UV,epsilon);
      SlU = new T[Nl*UV.r];
      VSr = new T[Nr*UV.r];
      gemTm(1.,Sl,UV.U,0.,SlU,Nl,r,UV.r);
      gemm (1.,UV.V,Sr,0.,VSr,UV.r,r,Nr);
      std::free(A);
      std::free(Sl);
      std::free(Sr);
      std::free(UV.U);
      std::free(UV.V);
    }

    friend void gemm(lits<T,FLT,DIM,KRNL,ITYPE>& A, T* B, T* C, int nrhs){
      T* tmp0 = new T[A.UV.r*nrhs];
      gemm(1.,A.VSr,B,   0., tmp0,A.UV.r, A.Nr  ,nrhs);
      gemm(1.,A.SlU,tmp0,0., C   ,A.Nl  , A.UV.r,nrhs);
    }

    friend int Rank(lits<T,FLT,DIM,KRNL,ITYPE>& A){return A.UV.r;}

  }; // lits

}// THEIA
#endif
