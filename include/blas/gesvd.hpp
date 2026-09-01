//===================================================================
//
// Authors: Igor Chollet, Pierre Marchand
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
//  along with defmm.  If not, see <http://www.gnu.org/licenses/>
//
//====================================================================
#ifndef THEIA_BLAS_GESVD_HPP
#define THEIA_BLAS_GESVD_HPP
#include "./blas_parser_consts.hpp"
#include "../matrices/lrmat.hpp"

extern "C"{
  void sgesvd_(const char*, const char*, const int*, const int*, BLAS_S*, const int*, BLAS_S*, BLAS_S*, const int*, BLAS_S*, const int*, BLAS_S*, const int*,  int*);
  void dgesvd_(const char*, const char*, const int*, const int*, BLAS_D*, const int*, BLAS_D*, BLAS_D*, const int*, BLAS_D*, const int*, BLAS_D*, const int*,  int*);
  void cgesvd_(const char*, const char*, const int*, const int*, BLAS_C*, const int*, BLAS_S*, BLAS_C*, const int*, BLAS_C*, const int*, BLAS_C*, const int*, BLAS_S*, int*);
  void zgesvd_(const char*, const char*, const int*, const int*, BLAS_Z*, const int*, BLAS_D*, BLAS_Z*, const int*, BLAS_Z*, const int*, BLAS_Z*, const int*, BLAS_D*, int*);
}

namespace theia{
    
  inline void gesvd( BLAS_S *A, BLAS_S *U, BLAS_S *s, BLAS_S *V, int NbRow, int NbCol){
    int n = NbRow; int m = NbCol; int INF; int minMN = (n<m ? n : m);
    int nwk  = 5*minMN+(n>m ? n : m);
    BLAS_S      *wk   = new BLAS_S[nwk];
    int ldvt = (minMN<n ? minMN : n);
    sgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, &INF);
    delete [] wk;
  }
  inline void gesvd( BLAS_D *A, BLAS_D *U, BLAS_D *s, BLAS_D *V, int NbRow, int NbCol){
    int n = NbRow; int m = NbCol; int INF; int minMN = (n<m ? n : m);
    int nwk  = 5*minMN+(n>m ? n : m);
    BLAS_D      *wk   = new BLAS_D[nwk];
    int ldvt = (minMN<n ? minMN : n);
    dgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, &INF);
    delete [] wk;
  } 
  inline void gesvd( BLAS_C *A, BLAS_C *U, BLAS_S *s, BLAS_C *V, int NbRow, int NbCol){
    int n = NbRow; int m = NbCol; int INF; int minMN = (n<m ? n : m);
    BLAS_S* rwk = new BLAS_S[5*minMN];
    int nwk = 2*minMN+(n>m ? n : m);
    BLAS_C* wk  = new BLAS_C[nwk];
    int ldvt = (minMN<n ? minMN : n);
    cgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, rwk, &INF);
    delete [] rwk; delete [] wk;
  }
  inline void gesvd( BLAS_Z *A, BLAS_Z *U, BLAS_D *s, BLAS_Z *V, int NbRow, int NbCol){
    int n = NbRow; int m = NbCol; int INF; int minMN = (n<m ? n : m);
    BLAS_D* rwk = new BLAS_D[5*minMN];
    int nwk = 2*minMN+(n>m ? n : m);
    BLAS_Z* wk  = new BLAS_Z[nwk];
    int ldvt = (minMN<n ? minMN : n);
    zgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, rwk, &INF);
    delete [] rwk; delete [] wk;
  }

  
  // Only for square matrices
  template<typename FLT>
  inline void gesvd(FLT *A, int NbRowAndCol, lrmat<FLT>& lowrank, double& epsilon){
    FLT *S = new FLT[NbRowAndCol];
    FLT *U = new FLT[NbRowAndCol*NbRowAndCol];
    FLT *V = new FLT[NbRowAndCol*NbRowAndCol];
    gesvd(A,U,S,V,NbRowAndCol,NbRowAndCol);
    lowrank.m = NbRowAndCol;
    lowrank.n = NbRowAndCol;
    lowrank.r = 0;
    for(int u = 0; u < NbRowAndCol; u++){
      if(std::abs(S[u])/std::abs(S[0]) > epsilon){
        lowrank.r++;
      }
    }
    lowrank.U = new FLT[lowrank.r*NbRowAndCol];
    lowrank.V = new FLT[lowrank.r*NbRowAndCol];
    for(int i = 0; i < NbRowAndCol; i++){
      for(int j = 0; j < lowrank.r; j++){
	lowrank.U[i + j*NbRowAndCol] = U[i + j*NbRowAndCol];
      }
    }
    for(int i = 0; i < lowrank.r; i++){
      for(int j = 0; j < NbRowAndCol; j++){
	lowrank.V[i + j*lowrank.r] = S[i] * V[i + j*NbRowAndCol];
      }
    }
  }

  template<typename FLT>
  inline void gesvd(std::complex<FLT> *A, int NbRowAndCol, lrmat<std::complex<FLT> >& lowrank, double& epsilon){
    FLT               *S = new              FLT [NbRowAndCol];
    std::complex<FLT> *U = new std::complex<FLT>[NbRowAndCol*NbRowAndCol];
    std::complex<FLT> *V = new std::complex<FLT>[NbRowAndCol*NbRowAndCol];
    gesvd(A,U,S,V,NbRowAndCol,NbRowAndCol);
    lowrank.m = NbRowAndCol;
    lowrank.n = NbRowAndCol;
    lowrank.r = 0;
    for(int u = 0; u < NbRowAndCol; u++){
      if(std::abs(S[u])/std::abs(S[0]) > epsilon){
        lowrank.r++;
      }
    }
    lowrank.U = new std::complex<FLT>[lowrank.r*NbRowAndCol];
    lowrank.V = new std::complex<FLT>[lowrank.r*NbRowAndCol];
    for(int i = 0; i < NbRowAndCol; i++){
      for(int j = 0; j < lowrank.r; j++){
	lowrank.U[i + j*NbRowAndCol] = U[i + j*NbRowAndCol];
      }
    }
    for(int i = 0; i < lowrank.r; i++){
      for(int j = 0; j < NbRowAndCol; j++){
	lowrank.V[i + j*lowrank.r] = S[i] * V[i + j*NbRowAndCol];
      }
    }
  }
  template<>
  inline void gesvd<std::complex<float > >(std::complex<float> *A, int NbRowAndCol, lrmat<std::complex<float> >& lowrank, double& epsilon){
    gesvd<float>(A,NbRowAndCol,lowrank,epsilon);}
  template<>
  inline void gesvd<std::complex<double> >(std::complex<double> *A, int NbRowAndCol, lrmat<std::complex<double> >& lowrank, double& epsilon){
    gesvd<double>(A,NbRowAndCol,lowrank,epsilon);}


  //////////////////////////////
  ///// General interface
  /////
  ///// (should be prefered in future codes)
  /////
  template<typename T> struct artihemtic_precision{using type = T;};
  template<typename T> struct artihemtic_precision<std::complex<T>>{using type = T;};  
  template<typename T>
  inline void gesvd(T *A, int M, int N, double epsilon, T*& _U, T*& _V, int& r,
		    bool left_multiply        = false,
		    bool erase_matrix_A       = true){
    // Add float/double type in case of complex
    using FLT = typename artihemtic_precision<T>::type;
    // Copy matrix if user does not want it to be erased
    T* _A;
    if(erase_matrix_A){_A = A;}else{
      _A = new T[M*N];
      for(int i = 0; i < M*N; i++){_A[i] = A[i];}
    }
    // Get SVD in buffers
    int max_rank = std::min(M,N);
    FLT *S = new T  [max_rank];
    T   *U = new FLT[M*max_rank];
    T   *V = new T  [N*max_rank];
    gesvd(_A,U,S,V,M,N);
    // Copy result in outputs
    r = 0;
    for(int u = 0; u < max_rank; u++){if(std::abs(S[u])/std::abs(S[0]) > epsilon){r++;}}
    _U = new T[r*M];
    _V = new T[r*N];
    if(left_multiply){
      for (int j = 0; j < r; j++){
	for (int i = 0; i < M; i++){
	  _U[j*M + i] = U[j*M + i] * S[j];}}
      for (int j = 0; j < N; j++){
	for (int i = 0; i < r; i++){
	  _V[j*r + i] = V[j*max_rank + i];}}
    }else{
      for (int j = 0; j < r; j++){
	for (int i = 0; i < M; i++){
	  _U[j*M + i] = U[j*M + i];}}
      for (int j = 0; j < N; j++){
	for (int i = 0; i < r; i++){
	  _V[j*r + i] = V[j*max_rank + i] * S[i];}}
    }
    // Clean space
    delete [] S;
    delete [] U;
    delete [] V;
    if(!erase_matrix_A){delete [] _A;}
  }

  //////////////////////////////
  ///// Fixed rank
  
  // Only for square matrices
  template<typename FLT>
  inline void gesvd_fixed_rank(FLT *A, int NbRowAndCol, lrmat<FLT>& lowrank, int rank){
    FLT *S = new FLT[NbRowAndCol];
    FLT *U = new FLT[NbRowAndCol*NbRowAndCol];
    FLT *V = new FLT[NbRowAndCol*NbRowAndCol];
    gesvd(A,U,S,V,NbRowAndCol,NbRowAndCol);
    lowrank.m = NbRowAndCol;
    lowrank.n = NbRowAndCol;
    lowrank.r = rank;
    lowrank.U = new FLT[lowrank.r*NbRowAndCol];
    lowrank.V = new FLT[lowrank.r*NbRowAndCol];
    for(int i = 0; i < NbRowAndCol; i++){
      for(int j = 0; j < lowrank.r; j++){
	lowrank.U[i + j*NbRowAndCol] = U[i + j*NbRowAndCol];
      }
    }
    for(int i = 0; i < lowrank.r; i++){
      for(int j = 0; j < NbRowAndCol; j++){
	lowrank.V[i + j*lowrank.r] = S[i] * V[i + j*NbRowAndCol];
      }
    }
  }

  template<typename FLT>
  inline void gesvd_fixed_rank(std::complex<FLT> *A, int NbRowAndCol, lrmat<std::complex<FLT> >& lowrank, int rank){
    FLT               *S = new              FLT [NbRowAndCol];
    std::complex<FLT> *U = new std::complex<FLT>[NbRowAndCol*NbRowAndCol];
    std::complex<FLT> *V = new std::complex<FLT>[NbRowAndCol*NbRowAndCol];
    gesvd(A,U,S,V,NbRowAndCol,NbRowAndCol);
    lowrank.m = NbRowAndCol;
    lowrank.n = NbRowAndCol;
    lowrank.r = rank;
    lowrank.U = new std::complex<FLT>[lowrank.r*NbRowAndCol];
    lowrank.V = new std::complex<FLT>[lowrank.r*NbRowAndCol];
    for(int i = 0; i < NbRowAndCol; i++){
      for(int j = 0; j < lowrank.r; j++){
	lowrank.U[i + j*NbRowAndCol] = U[i + j*NbRowAndCol];
      }
    }
    for(int i = 0; i < lowrank.r; i++){
      for(int j = 0; j < NbRowAndCol; j++){
	lowrank.V[i + j*lowrank.r] = S[i] * V[i + j*NbRowAndCol];
      }
    }
  }
  template<>
  inline void gesvd_fixed_rank<std::complex<float > >(std::complex<float> *A, int NbRowAndCol, lrmat<std::complex<float> >& lowrank, int rank){
    gesvd_fixed_rank<float>(A,NbRowAndCol,lowrank,rank);}
  template<>
  inline void gesvd_fixed_rank<std::complex<double> >(std::complex<double> *A, int NbRowAndCol, lrmat<std::complex<double> >& lowrank, int rank){
    gesvd_fixed_rank<double>(A,NbRowAndCol,lowrank,rank);}
  //////////////////////////////

  
} // THEIA

#endif
