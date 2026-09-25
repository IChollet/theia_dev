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
#include <vector>
#include "./polynomials.hpp"

namespace theia{
  
  inline int prod_of_orders(const int* L, int dim){int res = 1; for(int k = 0; k < dim; k++){res *= L[k];} return res;}

  /*
    Tensor grid of interpolation nodes in the box [mins,maxs],
    with L[k] nodes along dimension k (last dimension fastest)
  */
  template<int DIM, typename FLT, int ITYPE>
  inline void get_multivariate_interp_nodes(const int* L,
					    const FLT* mins, const FLT* maxs,
					    std::array<FLT,DIM>* z){
    int Ld = prod_of_orders(L,DIM);
    for(int i = 0; i < Ld; i++){
      int itmp = i;
      for(int k = DIM-1; k >= 0; k--){
	int ki  = (itmp % L[k]);
	itmp    = (itmp / L[k]);
	z[i][k] = get_node<ITYPE>(ki,L[k],mins[k],maxs[k]);
      }
    }
  }

  // Same order L in every dimension
  template<int DIM, typename FLT, int ITYPE>
  inline void get_multivariate_interp_nodes(int L,
					    FLT* mins, FLT* maxs,
					    std::array<FLT,DIM>* z){
    int Ls[DIM];
    for(int k = 0; k < DIM; k++){Ls[k] = L;}
    get_multivariate_interp_nodes<DIM,FLT,ITYPE>(Ls,mins,maxs,z);
  }

  /*
    Multivariate Lagrange polynomials evaluated at the N particles prts:
    S (prod L x N, column major, allocated if nullptr) with L[k] polynomials
    along dimension k (last dimension fastest)
  */
  template<int DIM, typename FLT, typename T, int ITYPE>
  inline void get_polynomials(const int* L, T*& S,
			      const FLT* mins, const FLT* maxs,
			      std::array<FLT,DIM>* prts, int N){
    int Ld  = prod_of_orders(L,DIM);
    if(S == nullptr)
      S     = new T   [Ld*N];
    FLT **V = new FLT*[DIM];
    for(int k = 0; k < DIM; k++){
      int Lk  = L[k];
      V[k] = new FLT[Lk*N];
      FLT* Vk = V[k];
      FLT ctr = mins[k]+maxs[k];
      ctr    *= 0.5;
      FLT rad = std::abs(ctr-maxs[k]);
      if(rad == FLT(0.)){
	// Flat box along k: constant polynomials 1/L (partition of unity)
	for(int j = 0; j < Lk*N; j++){Vk[j] = FLT(1.)/FLT(Lk);}
	continue;
      }
      for(int j = 0; j < N; j++){
	FLT y     = (prts[j][k]-ctr)/rad;
        for(int i = 0; i < Lk; i++){
	  Vk[j*Lk+i] = C1D<ITYPE>(y,i,Lk);
	}
      }
    }
    // Row of particle j built by successive Kronecker products
    //   S_j = V_0(j,:) x V_1(j,:) x ... x V_{DIM-1}(j,:)   (last dimension fastest)
    // computed in place, from the end so that s[a] is read before being overwritten
    for(int j = 0; j < N; j++){
      T* s = S + j*Ld;
      s[0] = T(1.);
      int n = 1;
      for(int k = 0; k < DIM; k++){
	int Lk = L[k];
	const FLT* v = V[k] + j*Lk;
	for(int a = n-1; a >= 0; a--){
	  T sa = s[a];
	  T* sL = s + a*Lk;
	  for(int i = 0; i < Lk; i++){sL[i] = sa * v[i];}
	}
	n *= Lk;
      }
    }
    for(int k = 0; k < DIM; k++){delete [] V[k];}
    delete [] V;
  }

  // Same order L in every dimension
  template<int DIM, typename FLT, typename T, int ITYPE>
  inline void get_polynomials(int L, T*& S,
			      const FLT* mins, const FLT* maxs,
			      std::array<FLT,DIM>* prts, int N){
    int Ls[DIM];
    for(int k = 0; k < DIM; k++){Ls[k] = L;}
    get_polynomials<DIM,FLT,T,ITYPE>(Ls,S,mins,maxs,prts,N);
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
    
    int     L;                        // Interpolation order
    std::vector<FLT> minsl;           // Left    lower interval bounds
    std::vector<FLT> maxsl;           // Left    maximal interval bounds
    int     Nl;                       // Left    number of particles
    std::array<FLT,DIM>* prtsl;       // Left    particles
    std::vector<FLT> minsr;           // Right   [...]
    std::vector<FLT> maxsr;           // Right   [...]
    int     Nr;                       // Right   [...]
    std::array<FLT,DIM>* prtsr;       // Right   [...]
    int    r;                         // Left and right ranks
    KRNL*  K;                         // Kernel reference
    std::vector<T> SlU;               // Final left term 
    std::vector<T> VSr;               // Final right term
    int    rank_of_compressed_matrix; // Rank of the compressed output matrix 

    // Interpolation nodes, central symbolic matrix A (Ld x Ld) and
    // left/right polynomials Sl/Sr (allocated here, freed by the caller)
    void get_interpolation(T*& A, T*& Sl, T*& Sr){
      int Ld  = myintpow(L,DIM);
      A       = new T  [Ld*Ld];
      Sl      = nullptr;
      Sr      = nullptr;
      std::array<FLT,DIM> *px = new std::array<FLT,DIM>[Ld];
      std::array<FLT,DIM> *py = new std::array<FLT,DIM>[Ld];
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsr.data(),maxsr.data(),py);
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsl.data(),maxsl.data(),px);
      get_symbolic_matrix<DIM,FLT,T,KRNL>(px,py,Ld,Ld,A,K);
      get_polynomials<DIM,FLT,T,ITYPE>(L,Sr,minsr.data(),maxsr.data(),prtsr,Nr);
      get_polynomials<DIM,FLT,T,ITYPE>(L,Sl,minsl.data(),maxsl.data(),prtsl,Nl);
      delete [] px;
      delete [] py;
    }

    // SlU = Sl^T U and VSr = V Sr (outputs allocated here), then frees inputs
    void apply_polynomials(T* A, T* Sl, T* Sr, lrmat<T>& UV, int Ld,
			   T*& _SlU, T*& _VSr){
      _SlU = new T[Nl*rank_of_compressed_matrix];
      _VSr = new T[Nr*rank_of_compressed_matrix];
      gemTm(1.,Sl,UV.U,0.,_SlU,Nl,Ld,rank_of_compressed_matrix);
      gemm (1.,UV.V,Sr,0.,_VSr,rank_of_compressed_matrix,Ld,Nr);
      delete [] A;
      delete [] Sl;
      delete [] Sr;
      delete [] UV.U;
      delete [] UV.V;
    }
    
  public:
    
    lits(FLT* minsl_, FLT* maxsl_, std::array<FLT,DIM>* prtsl_, int Nl_,
	 FLT* minsr_, FLT* maxsr_, std::array<FLT,DIM>* prtsr_, int Nr_,
	 int L_, KRNL* K_){
      L = L_; prtsl = prtsl_; Nl = Nl_; K = K_;
      minsl.assign(minsl_, minsl_+DIM);
      maxsl.assign(maxsl_, maxsl_+DIM);
      prtsr = prtsr_; Nr = Nr_;
      minsr.assign(minsr_, minsr_+DIM);
      maxsr.assign(maxsr_, maxsr_+DIM);
      rank_of_compressed_matrix = -1;
    }
    
    void get_source_nodes(std::array<FLT,DIM>*& py, int& number_of_nodes){
      number_of_nodes = myintpow(L,DIM);
      py = new std::array<FLT,DIM>[number_of_nodes];
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsr.data(),maxsr.data(),py);}
    
    void get_target_nodes(std::array<FLT,DIM>*& px, int& number_of_nodes){
      number_of_nodes = myintpow(L,DIM);
      px = new std::array<FLT,DIM>[number_of_nodes];
      get_multivariate_interp_nodes<DIM,FLT,ITYPE>(L,minsl.data(),maxsl.data(),px);}
    
    void get_UV(double epsilon){
      T *_SlU, *_VSr;
      int rank;
      get_UV(epsilon, _SlU, _VSr, rank);
      SlU.assign(_SlU, _SlU + Nl*rank);
      VSr.assign(_VSr, _VSr + Nr*rank);
      delete [] _SlU;
      delete [] _VSr;
    }

    friend void gemm(lits<T,FLT,DIM,KRNL,ITYPE>& A, T* B, T* C, int nrhs){
      T* tmp0 = new T[A.rank_of_compressed_matrix*nrhs];
      gemm(1.,A.VSr.data(),B,0.,tmp0,A.rank_of_compressed_matrix,A.Nr,nrhs);
      gemm(1.,A.SlU.data(),tmp0,0., C,A.Nl, A.rank_of_compressed_matrix,nrhs);
      delete [] tmp0;
    }
    
    /*
      Get interpolation + SVD compression with output arrays given by user.
      /!\ Arrays are allocated inside this finction /!\
      Rank of returned matrix is also returned in argument "rank"
    */
    void get_UV(double epsilon, T*& _SlU, T*& _VSr, int& rank){
      T *A, *Sl, *Sr;
      lrmat<T> UV;
      r = myintpow(L,DIM);
      get_interpolation(A,Sl,Sr);
      gesvd<T>(A,r,UV,epsilon);
      //paca<T>(A,r,UV,epsilon);
      rank_of_compressed_matrix = UV.r;
      apply_polynomials(A,Sl,Sr,UV,r,_SlU,_VSr);
      rank = rank_of_compressed_matrix;
    }

    /* Here, the rank is fixed */
    void get_UV(int rank, T*& _SlU, T*& _VSr){
      int Ld  = myintpow(L,DIM);
      if(rank > Ld){std::cout << "Required rank is higher than possible one using interpolation" << std::endl; exit(1);}
      T *A, *Sl, *Sr;
      lrmat<T> UV;
      get_interpolation(A,Sl,Sr);
      gesvd_fixed_rank<T>(A,Ld,UV,rank);
      rank_of_compressed_matrix = rank;
      apply_polynomials(A,Sl,Sr,UV,Ld,_SlU,_VSr);
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
