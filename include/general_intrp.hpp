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
    if(S == nullptr)
      S     = new T   [Ld*N];
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

  /*
    mins  : min bounds of cell (FLT*)
    maxs  : max bounds of cell (FLT*)
    N     : number of particles (int)
    py    : particle array (std::array<FLT,DIM>*)
    L     : interpolation order (int)
    A     : P2M matrix (FLT*)
  */
  template<int DIM, typename T, typename FLT, int ITYPE>
  inline void get_P2M(FLT* mins, FLT* maxs,
		      int N, std::array<FLT,DIM>* py,
		      int L, T* A){
    get_polynomials<DIM,FLT,T,ITYPE>(L,A,mins,maxs,py,N);
  }

  /*
  template<int DIM, typename T, typename FLT, int I_TYPE>
  inline void get_M2L(FLT* minsX, FLT* maxsX, int Lx,
		      FLT* minsY, FLT* maxsY, int Ly,
		      T* U, T* V, int& rank){
    int Lxd  = myintpow(Lx,DIM);
    int Lyd  = myintpow(Lx,DIM);
    A        = new T  [Lxd*Lyd];
    std::array<FLT,DIM> *px = new std::array<FLT,DIM>[Lxd];
    std::array<FLT,DIM> *py = new std::array<FLT,DIM>[Lyd];
    get_multivariate_interp_nodes<DIM,FLT,ITYPE>(Lx,minsr,maxsr,py);
    get_multivariate_interp_nodes<DIM,FLT,ITYPE>(Ly,minsl,maxsl,px);
    get_symbolic_matrix<DIM,FLT,T,KRNL>(px,py,Lxd,Lyd,A,K);
    lrmat<T> UV;
    gesvd<T>(A,r,UV,epsilon);
    rank = UV.r;
    for(int i = 0; i < Lxd; i++){
      for(int k = 0; k < rank; k++){
      }
    }
  }
  */

  template<int DIM, typename FLT, int LEFT_TYPE, int RIGHT_TYPE>
  inline void get_M2M(int * left_L, FLT*  left_mins, FLT*  left_maxs,
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
    A.set(S,left_L,right_L);
  }

  // AJOUTER une fonction qui prend en argument une liste de cellules filles et une cellule mère, et qui calcule la matrice de réinterpolation sur la mère (structure de liste de prod de kron)
  
  // Templates :
  //     T     : Kernel type
  //     FLT   : Floating point type (i.e. Lagrange pol. type)
  //     DIM   : Ambiant dimension
  //     KRNL  : Kernel class type
  //     ITYPE : Interpolation type (0 for Chebyshev / 1 for equispaced)
  template<typename T, typename FLT, int DIM, class KRNL, int ITYPE>  class lits{
  private:
    
    T*      Sl = nullptr;             // Left    polynomials
    T*      Sr = nullptr;             // Right   polynomials
    T*      A = nullptr;              // Central symbolic matrix
    int     L;                        // Interpolation order
    FLT*    minsl;                    // Left    lower interval bounds
    FLT*    maxsl;                    // Left    maximal interval bounds
    int     Nl;                       // Left    number of particles
    std::array<FLT,DIM>* prtsl;       // Left    particles
    FLT*    minsr;                    // Right   [...]
    FLT*    maxsr;                    // Right   [...]
    int     Nr;                       // Right   [...]
    std::array<FLT,DIM>* prtsr;       // Right   [...]
    int    r;                         // Left and right ranks
    lrmat<T> UV;                      // Low-rank factorization of A
    KRNL*  K;                         // Kernel reference
    T*     SlU = nullptr;             // Final left term 
    T*     VSr = nullptr;             // Final right term
    int    rank_of_compressed_matrix; // Rank of the compressed output matrix 
    
  public:
    
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
      rank_of_compressed_matrix = -1;
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
      rank_of_compressed_matrix = UV.r;
      SlU = new T[Nl*rank_of_compressed_matrix];
      VSr = new T[Nr*rank_of_compressed_matrix];
      gemTm(1.,Sl,UV.U,0.,SlU,Nl,r,rank_of_compressed_matrix);
      gemm (1.,UV.V,Sr,0.,VSr,rank_of_compressed_matrix,r,Nr);
      std::free(A);
      std::free(Sl);
      std::free(Sr);
      std::free(UV.U);
      std::free(UV.V);
    }

    friend void gemm(lits<T,FLT,DIM,KRNL,ITYPE>& A, T* B, T* C, int nrhs){
      T* tmp0 = new T[A.rank_of_compressed_matrix*nrhs];
      gemm(1.,A.VSr,B,0.,tmp0,A.rank_of_compressed_matrix,A.Nr,nrhs);
      gemm(1.,A.SlU,tmp0,0., C,A.Nl, A.rank_of_compressed_matrix,nrhs);
    }
    
    /*
      Get interpolation + SVD compression with output arrays given by user.
      /!\ Arrays are allocated inside this finction /!\
      Rank of returned matrix is also returned in argument "rank"
    */
    void get_UV(double epsilon, T*& _SlU, T*& _VSr, int& rank){
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
      rank_of_compressed_matrix = UV.r;
      _SlU = new T[Nl*rank_of_compressed_matrix];
      _VSr = new T[Nr*rank_of_compressed_matrix];
      gemTm(1.,Sl,UV.U,0.,_SlU,Nl,r,rank_of_compressed_matrix);
      gemm (1.,UV.V,Sr,0.,_VSr,rank_of_compressed_matrix,r,Nr);
      rank = rank_of_compressed_matrix;
      std::free(A);
      std::free(Sl);
      std::free(Sr);
      std::free(UV.U);
      std::free(UV.V);
    }

    /* Here, the rank is fixed */
    void get_UV(int rank, T*& _SlU, T*& _VSr){
      int Ld  = myintpow(L,DIM);
      if(rank > Ld){std::cout << "Required rank is higher than possible one using interpolation" << std::endl; exit(1);}
      A       = new T  [Ld*Ld];
      std::array<FLT,DIM> *px = new std::array<FLT,DIM>[Ld];
      std::array<FLT,DIM> *py = new std::array<FLT,DIM>[Ld];
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsr,maxsr,py);
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsl,maxsl,px);
      get_symbolic_matrix<DIM,FLT,T,KRNL>(px,py,Ld,Ld,A,K);
      get_polynomials<DIM,FLT,T,ITYPE>(L,Sr,minsr,maxsr,prtsr,Nr);
      get_polynomials<DIM,FLT,T,ITYPE>(L,Sl,minsl,maxsl,prtsl,Nl);
      gesvd_fixed_rank<T>(A,Ld,UV,rank);
      rank_of_compressed_matrix = rank;
      _SlU = new T[Nl*rank_of_compressed_matrix];
      _VSr = new T[Nr*rank_of_compressed_matrix];
      gemTm(1.,Sl,UV.U,0.,_SlU,Nl,Ld,rank_of_compressed_matrix);
      gemm (1.,UV.V,Sr,0.,_VSr,rank_of_compressed_matrix,Ld,Nr);
      std::free(A);
      std::free(Sl);
      std::free(Sr);
      std::free(UV.U);
      std::free(UV.V);
    }

    friend int Rank(lits<T,FLT,DIM,KRNL,ITYPE>& A){return A.rank_of_compressed_matrix;}

  }; // lits
  
  template<typename T, typename FLT, int DIM, class KRNL>
  void get_lits_cheb(FLT* minsl_, FLT* maxsl_, std::array<FLT,DIM>* prtsl_, int Nl_,
		     FLT* minsr_, FLT* maxsr_, std::array<FLT,DIM>* prtsr_, int Nr_,
		     int L_, KRNL* K_, double epsilon, T*& _SlU, T*& _VSr, int& rank){
    lits<T,FLT,DIM,KRNL,0> GL(minsl_,maxsl_,prtsl_,Nl_,
			      minsr_,maxsr_,prtsr_,Nr_,
			      L_, K_);
    GL.get_UV(epsilon, _SlU, _VSr, rank);
  }

  template<typename T, typename FLT, int DIM, class KRNL>
  void get_lits_cheb_fixed_rank(FLT* minsl_, FLT* maxsl_, std::array<FLT,DIM>* prtsl_, int Nl_,
				FLT* minsr_, FLT* maxsr_, std::array<FLT,DIM>* prtsr_, int Nr_,
				int L_, KRNL* K_, double epsilon, T*& _SlU, T*& _VSr, int rank){
    lits<T,FLT,DIM,KRNL,0> GL(minsl_,maxsl_,prtsl_,Nl_,
			      minsr_,maxsr_,prtsr_,Nr_,
			      L_, K_);
    GL.get_UV(rank, _SlU, _VSr);
  }

}// THEIA
#endif
