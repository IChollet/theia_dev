#ifndef THEIA_SKP_NKMM_HPP
#define THEIA_SKP_NKMM_HPP

#include "./matrices.hpp"
#include <cmath>
#include <iostream>

namespace theia{

  template<typename T>
  int array_prod(T* a, int start, int end){
    int res = 1;
    for(int i = start; i < end; i++){
      res *= a[i];
    }
    return res;
  }

  template<typename T>
  int max_in_array(T* a, int start, int end){
    T res = a[start];
    for(int i = start+1; i < end; i++){
      res = std::max(res,a[i]);
    }
    return res;
  }

  inline void i2I(int i, int* I, int *N, int dim){
    int _i = i;
    for(int k = 0; k < dim; k++){
      I[dim-1-k] = (_i%N[dim-1-k]);
      _i        /=     N[dim-1-k] ;
    }
  }


  /*
    Naive Kronecker Matrix-vector product
  */
  template<typename FLT>
  void nkmv(FLT** A, int* nA, int* mA,
	    FLT*  q, FLT*  p, int dim){
    int n = array_prod(nA, 0, dim);
    int m = array_prod(mA, 0, dim);
    int nm = n*m;
    FLT *Z = new FLT[nm];
    int *I = new int[dim];
    int *J = new int[dim];
    for(int i = 0; i < n; i++){
      i2I(i,I,nA,dim);
      for(int j = 0; j < m; j++){
        i2I(j,J,mA,dim);
	Z[i+j*n] = FLT(1.);
        for(int k = 0; k < dim; k++){
	  Z[i+j*n] *= A[k][I[k]+J[k]*nA[k]];
        }
      }
    }
    theia::gemm(1., Z, q, 0., p, n, m, 1);
    delete [] Z;
    delete [] I;
    delete [] J;
  }

  /*
    Naive Kronecker product with transposed matrix
  */
  template<typename FLT>
  void nkmTv(FLT** A, int* nA, int* mA,
	     FLT*  q, FLT*  p, int dim){
    int n = array_prod(mA, 0, dim);
    int m = array_prod(nA, 0, dim);
    int nm = n*m;
    FLT *Z = new FLT[nm];
    int *I = new int[dim];
    int *J = new int[dim];
    for(int i = 0; i < n; i++){
      i2I(i,I,mA,dim);
      for(int j = 0; j < m; j++){
        i2I(j,J,nA,dim);
	Z[i+j*n] = FLT(1.);
        for(int k = 0; k < dim; k++){
	  Z[i+j*n] *= A[k][I[k]*nA[k]+J[k]];
        }
      }
    }
    theia::gemm(1., Z, q, 0., p, n, m, 1);    
    delete [] Z;
    delete [] I;
    delete [] J;
  }
  
} // THEIA

#endif
