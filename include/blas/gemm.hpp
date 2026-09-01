#ifndef THEIA_BLAS_GEMM_HPP
#define THEIA_BLAS_GEMM_HPP
#include "./blas_parser_consts.hpp"
#include "../matrices/lrmat.hpp"

extern "C"{
  void sgemm_(const char*, const char*, const int*, const int*, const int*, const BLAS_S*, const BLAS_S*, const int*, const BLAS_S*, const int*, const BLAS_S*, BLAS_S*, const int*);
  void dgemm_(const char*, const char*, const int*, const int*, const int*, const BLAS_D*, const BLAS_D*, const int*, const BLAS_D*, const int*, const BLAS_D*, BLAS_D*, const int*);
  void cgemm_(const char*, const char*, const int*, const int*, const int*, const BLAS_C*, const BLAS_C*, const int*, const BLAS_C*, const int*, const BLAS_C*, BLAS_C*, const int*);
  void zgemm_(const char*, const char*, const int*, const int*, const int*, const BLAS_Z*, const BLAS_Z*, const int*, const BLAS_Z*, const int*, const BLAS_Z*, BLAS_Z*, const int*);
  void sgemv_(const char*, const int*, const int*, const BLAS_S*, const BLAS_S*, const int*, const BLAS_S*, const int*, const BLAS_S*, BLAS_S*, const int*);
  void dgemv_(const char*, const int*, const int*, const BLAS_D*, const BLAS_D*, const int*, const BLAS_D*, const int*, const BLAS_D*, BLAS_D*, const int*);
  void cgemv_(const char*, const int*, const int*, const BLAS_C*, const BLAS_C*, const int*, const BLAS_C*, const int*, const BLAS_C*, BLAS_C*, const int*);
  void zgemv_(const char*, const int*, const int*, const BLAS_Z*, const BLAS_Z*, const int*, const BLAS_Z*, const int*, const BLAS_Z*, BLAS_Z*, const int*);
}

namespace theia{
  
  inline void gemm(BLAS_S u, BLAS_S* A, BLAS_S* B, BLAS_S v, BLAS_S* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    sgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemm(BLAS_D u, BLAS_D* A, BLAS_D* B, BLAS_D v, BLAS_D* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    dgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_S u, BLAS_S* A, BLAS_S* B, BLAS_S v, BLAS_S* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    sgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_D u, BLAS_D* A, BLAS_D* B, BLAS_D v, BLAS_D* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    dgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  inline void gemm(BLAS_C u, BLAS_C* A, BLAS_C* B, BLAS_C v, BLAS_C* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    cgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemm(BLAS_Z u, BLAS_Z* A, BLAS_Z* B, BLAS_Z v, BLAS_Z* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    zgemm_(charN,charN,&n,&s,&k,&u,A,&n,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_C u, BLAS_C* A, BLAS_C* B, BLAS_C v, BLAS_C* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    cgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  inline void gemTm(BLAS_Z u, BLAS_Z* A, BLAS_Z* B, BLAS_Z v, BLAS_Z* C, int _n, int _k, int _s){
    int n = _n;  int k = _k;  int s = _s;
    zgemm_(charT,charN,&n,&s,&k,&u,A,&k,B,&k, &v,C,&n);}
  
  inline void gemv(BLAS_S u, BLAS_S* A, BLAS_S* x, BLAS_S v, BLAS_S* y, int L, int K){int n = L; int s = K; sgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemv(BLAS_D u, BLAS_D* A, BLAS_D* x, BLAS_D v, BLAS_D* y, int L, int K){int n = L; int s = K; dgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemv(BLAS_C u, BLAS_C* A, BLAS_C* x, BLAS_C v, BLAS_C* y, int L, int K){int n = L; int s = K; cgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemv(BLAS_Z u, BLAS_Z* A, BLAS_Z* x, BLAS_Z v, BLAS_Z* y, int L, int K){int n = L; int s = K; zgemv_(charN, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_S u, BLAS_S* A, BLAS_S* x, BLAS_S v, BLAS_S* y, int L, int K){int n = L; int s = K; sgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_D u, BLAS_D* A, BLAS_D* x, BLAS_D v, BLAS_D* y, int L, int K){int n = L; int s = K; dgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_C u, BLAS_C* A, BLAS_C* x, BLAS_C v, BLAS_C* y, int L, int K){int n = L; int s = K; cgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}
  inline void gemTv(BLAS_Z u, BLAS_Z* A, BLAS_Z* x, BLAS_Z v, BLAS_Z* y, int L, int K){int n = L; int s = K; zgemv_(charT, &n, &s, &u, A, &n, x, &IN_ONE, &v, y, &IN_ONE);}

  template<typename FLT>
  inline void gemm(FLT u, lrmat<FLT>& UV, FLT* B, FLT v, FLT* C, int m, int n, int s){
    FLT* tmp = new FLT[UV.r*s];
    gemm(1., UV.V, B,   0., tmp, UV.r, UV.n, s);
    gemm(u , UV.U, tmp, v , C,   UV.m, UV.r, s);
  }  

}

#endif
