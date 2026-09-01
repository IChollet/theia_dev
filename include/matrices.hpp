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
#ifndef THEIA_MATRICES_HPP
#define THEIA_MATRICES_HPP

#include "./blas/gemm.hpp"
#include "./blas/gesvd.hpp"
#include "./blas/getrf.hpp"
#include "./blas/eigenvalues.hpp"

#include "./matrices/lrmat.hpp"

namespace theia{

  template<typename T> void copy(T* input, int size, T*& output){
    output = new T[size];
    for(int i = 0; i < size; i++){output[i] = input[i];}
  }
  
  template<typename T> void invert(T* A, int n){
    int* ipiv = new int[n];
    getrf(n, n, A, n, ipiv);
    getri(n, A, n, ipiv);
    delete[] ipiv;
  }

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
