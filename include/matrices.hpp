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
//  along with defmm.  If not, see <http://www.gnu.org/licenses/>
//
//====================================================================
#include <complex>
#include <vector>
#ifndef THEIA_MATRICES_HPP
#define THEIA_MATRICES_HPP
#ifndef BLAS_PARSER_CONSTS_THEIA
#define BLAS_PARSER_CONSTS_THEIA
#define BLAS_S float
#define BLAS_D double
#define BLAS_C std::complex<float>
#define BLAS_Z std::complex<double>
BLAS_S S_ZERO     =  0.0;
BLAS_S S_ONE      =  1.0;
BLAS_S S_MONE     = -1.0;
BLAS_D D_ZERO     =  0.0;
BLAS_D D_ONE      =  1.0;
BLAS_D D_MONE     = -1.0;
BLAS_C C_ZERO     =  0.0;
BLAS_C C_ONE      =  1.0;
BLAS_C C_MONE     = -1.0;
BLAS_Z Z_ZERO     =  0.0;
BLAS_Z Z_ONE      =  1.0;
BLAS_Z Z_MONE     = -1.0;
int    IN_ONE     =  1  ;
const char* charN = "N" ;
const char* charT = "T" ;
const char* charC = "C" ;
const char* charS = "S" ;
const char* charA = "A" ;

#endif

extern "C"{
  void sgemm_(const char*, const char*, unsigned*, unsigned*, unsigned*, BLAS_S*, BLAS_S*, 
	      unsigned*, BLAS_S*, unsigned*, BLAS_S*, BLAS_S*, unsigned*);
  void dgemm_(const char*, const char*, unsigned*, unsigned*, unsigned*, BLAS_D*, BLAS_D*, 
	      unsigned*, BLAS_D*, unsigned*, BLAS_D*, BLAS_D*, unsigned*);
  void cgemm_(const char*, const char*, unsigned*, unsigned*, unsigned*, BLAS_C*, BLAS_C*, 
	      unsigned*, BLAS_C*, unsigned*, BLAS_C*, BLAS_C*, unsigned*);
  void zgemm_(const char*, const char*, unsigned*, unsigned*, unsigned*, BLAS_Z*, BLAS_Z*, 
	      unsigned*, BLAS_Z*, unsigned*, BLAS_Z*, BLAS_Z*, unsigned*);  
  void sgemv_(const char*, int*, int*, BLAS_S*,BLAS_S*, int*, BLAS_S*, int*, BLAS_S*, BLAS_S*, int*);
  void dgemv_(const char*, int*, int*, BLAS_D*,BLAS_D*, int*, BLAS_D*, int*, BLAS_D*, BLAS_D*, int*);
  void cgemv_(const char*, int*, int*, BLAS_C*,BLAS_C*, int*, BLAS_C*, int*, BLAS_C*, BLAS_C*, int*);
  void zgemv_(const char*, int*, int*, BLAS_Z*,BLAS_Z*, int*, BLAS_Z*, int*, BLAS_Z*, BLAS_Z*, int*);
}
extern "C"{
  void sgesvd_(const char*, const char*, unsigned*, unsigned*, BLAS_S*, unsigned*, BLAS_S*, BLAS_S*, unsigned*, BLAS_S*, unsigned*, BLAS_S*, unsigned*, int*);
  void dgesvd_(const char*, const char*, unsigned*, unsigned*, BLAS_D*, unsigned*, BLAS_D*, BLAS_D*, unsigned*, BLAS_D*, unsigned*, BLAS_D*, unsigned*, int*);
  void cgesvd_(const char*, const char*, unsigned*, unsigned*, BLAS_C*, unsigned*, BLAS_S*, BLAS_C*, unsigned*, BLAS_C*, unsigned*, BLAS_C*, unsigned*, BLAS_C*,int*);
  void zgesvd_(const char*, const char*, unsigned*, unsigned*, BLAS_Z*, unsigned*, BLAS_D*, BLAS_Z*, unsigned*, BLAS_Z*, unsigned*, BLAS_Z*, unsigned*, BLAS_Z*,int*);
}

namespace theia{
  
  inline void gemm(BLAS_S u, BLAS_S* A, BLAS_S* B, BLAS_S v, BLAS_S* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    sgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemm(BLAS_D u, BLAS_D* A, BLAS_D* B, BLAS_D v, BLAS_D* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    dgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_S u, BLAS_S* A, BLAS_S* B, BLAS_S v, BLAS_S* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    sgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_D u, BLAS_D* A, BLAS_D* B, BLAS_D v, BLAS_D* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    dgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  inline void gemm(BLAS_C u, BLAS_C* A, BLAS_C* B, BLAS_C v, BLAS_C* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    cgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemm(BLAS_Z u, BLAS_Z* A, BLAS_Z* B, BLAS_Z v, BLAS_Z* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    zgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_C u, BLAS_C* A, BLAS_C* B, BLAS_C v, BLAS_C* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    cgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_Z u, BLAS_Z* A, BLAS_Z* B, BLAS_Z v, BLAS_Z* C, int _n, int _k, int _s){
    unsigned int n = _n;  unsigned int k = _k;  unsigned int s = _s;
    zgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  
  inline void gemv(BLAS_S u, BLAS_S* A, BLAS_S* x, BLAS_S v, BLAS_S* y, int L, int K){int n = L; int s = K; sgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemv(BLAS_D u, BLAS_D* A, BLAS_D* x, BLAS_D v, BLAS_D* y, int L, int K){int n = L; int s = K; dgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemv(BLAS_C u, BLAS_C* A, BLAS_C* x, BLAS_C v, BLAS_C* y, int L, int K){int n = L; int s = K; cgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemv(BLAS_Z u, BLAS_Z* A, BLAS_Z* x, BLAS_Z v, BLAS_Z* y, int L, int K){int n = L; int s = K; zgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_S u, BLAS_S* A, BLAS_S* x, BLAS_S v, BLAS_S* y, int L, int K){int n = L; int s = K; sgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_D u, BLAS_D* A, BLAS_D* x, BLAS_D v, BLAS_D* y, int L, int K){int n = L; int s = K; dgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_C u, BLAS_C* A, BLAS_C* x, BLAS_C v, BLAS_C* y, int L, int K){int n = L; int s = K; cgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_Z u, BLAS_Z* A, BLAS_Z* x, BLAS_Z v, BLAS_Z* y, int L, int K){int n = L; int s = K; zgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}

  inline void gesvd( BLAS_S *A, BLAS_S *U, BLAS_S *s, BLAS_S *V, int NbRow, int NbCol){
    unsigned int n = NbRow; unsigned int m = NbCol; int INF; int minMN = (n<m ? n : m);
    unsigned int nwk  = 5*minMN+(n>m ? n : m);
    BLAS_S      *wk   = new BLAS_S[nwk];
    unsigned int ldvt = (minMN<n ? minMN : n);
    sgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, &INF);
    delete [] wk;
  }
  inline void gesvd( BLAS_D *A, BLAS_D *U, BLAS_D *s, BLAS_D *V, int NbRow, int NbCol){
    unsigned int n = NbRow; unsigned int m = NbCol; int INF; int minMN = (n<m ? n : m);
    unsigned int nwk  = 5*minMN+(n>m ? n : m);
    BLAS_D      *wk   = new BLAS_D[nwk];
    unsigned int ldvt = (minMN<n ? minMN : n);
    dgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, &INF);
    delete [] wk;
  } 
  inline void gesvd( BLAS_C *A, BLAS_C *U, BLAS_S *s, BLAS_C *V, int NbRow, int NbCol){
    unsigned int n = NbRow; unsigned int m = NbCol; int INF; int minMN = (n<m ? n : m);
    BLAS_C* rwk = new BLAS_C[5*minMN];
    unsigned int nwk = 2*minMN+(n>m ? n : m);
    BLAS_C* wk  = new BLAS_C[nwk];
    unsigned int ldvt = (minMN<n ? minMN : n);
    cgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, rwk, &INF);
    delete [] rwk; delete [] wk;
  }
  inline void gesvd( BLAS_Z *A, BLAS_Z *U, BLAS_D *s, BLAS_Z *V, int NbRow, int NbCol){
    unsigned int n = NbRow; unsigned int m = NbCol; int INF; int minMN = (n<m ? n : m);
    BLAS_Z* rwk = new BLAS_Z[5*minMN];
    unsigned int nwk = 2*minMN+(n>m ? n : m);
    BLAS_Z* wk  = new BLAS_Z[nwk];
    unsigned int ldvt = (minMN<n ? minMN : n);
    zgesvd_(charS, charS, &n, &m, A, &n, s, U, &n, V, &ldvt, wk, &nwk, rwk, &INF);
    delete [] rwk; delete [] wk;
  }

  template<typename FLT>
  struct lrmat{
    int  m;   // Number of rows before factorization
    int  n;   // Number of colums before factorization
    int  r;   // Rank
    FLT* U;   // Left  side
    FLT* V;   // Right side 
  }; // LRMAT

  template<typename FLT>
  inline void allocate(lrmat<FLT>* UV, int m_, int n_, int r_){
    (*UV).m = m_;
    (*UV).n = n_;
    (*UV).r = r_;
    (*UV).U = (FLT*)std::malloc(sizeof(FLT)*m_*r_);
    (*UV).V = (FLT*)std::malloc(sizeof(FLT)*n_*r_);
  }

  template<typename FLT>
  inline void gemm(FLT u, lrmat<FLT>& UV, FLT* B, FLT v, FLT* C, int m, int n, int s){
    FLT* tmp = new FLT[UV.r*s];
    gemm(1., UV.V, B,   0., tmp, UV.r, UV.n, s);
    gemm(u , UV.U, tmp, v , C,   UV.m, UV.r, s);
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

  template<typename FLT>
  inline double nrm2(FLT* v, int n){
    double res = 0.;
    for(int i = 0; i < n; i++){
      double tmp = std::abs(v[i]);
      res += tmp*tmp;
    }
    return res;
  }

  template<typename FLT>
  inline int selectindex(FLT* v, int n){
    int    idx = -1;
    double rmx = -1;
    double tmp;
    for(int i = 0; i < n; i++){
      tmp = std::abs(v[i]);
      if(tmp > rmx){
	idx = i;
	rmx = tmp;
      }
    }
    return idx;
  }

  template<typename FLT>
  inline int selectindex(FLT* v, int n, std::vector<int>& alreadyvisited){
    int    idx = -1;
    double rmx = -1;
    double tmp;
    for(int i = 0; i < n; i++){
      tmp = std::abs(v[i]);
      if(tmp > rmx){
	bool b = false;
	for(int k = 0; k < alreadyvisited.size(); k++){
	  b |= (alreadyvisited[k] == i);
	}
	if(!b){
	  idx = i;
	  rmx = tmp;
	}
      }
    }
    return idx;
  }

  /*
    Partial Adaptive Cross Approximation of a square matrix of size m*m
  */
  template<typename FLT>
  void paca(FLT* X, int m, lrmat<FLT>& lowrank, double& epsilon){
    std::vector<int> alreadyvisitedrow;
    std::vector<int> alreadyvisitedcol;
    FLT *A = (FLT*)std::malloc(sizeof(FLT)*m*m);
    FLT *B = (FLT*)std::malloc(sizeof(FLT)*m*m);
    bool b = false;
    int I = 0, J = 0;
    int r = 0;
    for(int k = 0; k < m; k++){
      if(b){break;}
      alreadyvisitedrow.push_back(I);
      for(int j = 0; j < m; j++){
	B[j+k*m]  = X[I+j*m];
	for(int mu = 0; mu < k; mu++){
	  B[j+k*m] -= A[I+mu*m] * B[j+mu*m];
	}
      }
      J = selectindex<FLT>(B+k*m,m,alreadyvisitedcol);
      alreadyvisitedcol.push_back(J);
      for(int i = 0; i < m; i++){
	A[i+k*m]  = X[i+J*m];
	for(int mu = 0; mu < k; mu++){
	  A[i+k*m] -= A[i+mu*m] * B[J+mu*m];
	}
	A[i+k*m] /= B[J+k*m];
      }
      r++;
      I = selectindex<FLT>(A+k*m,m,alreadyvisitedrow);
      b = (nrm2(A+k*m,m)*nrm2(B+k*m,m) < epsilon*epsilon*nrm2(A,m)*nrm2(B,m));
    }
    lowrank.U = (FLT*)std::malloc(sizeof(FLT)*m*r);
    lowrank.V = (FLT*)std::malloc(sizeof(FLT)*m*r);
    lowrank.m = m;
    lowrank.n = m;
    lowrank.r = r;
    for(int i = 0; i < m; i++){
      for(int k = 0; k < r; k++){
	lowrank.U[i+k*m] = A[i+k*m];
	lowrank.V[k+i*r] = B[i+k*m];
      }
    }
    std::free(A);
    std::free(B);
  }

}// THEIA
#endif
