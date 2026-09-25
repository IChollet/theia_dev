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
#ifndef THEIA_TENSOR_P2M_HPP
#define THEIA_TENSOR_P2M_HPP
#include <vector>
#include "./kron.hpp"

namespace theia{

  /*
    Column-wise Kronecker (Khatri-Rao) product of DIM factors:
       column j = factors[0](:,j) x factors[1](:,j) x ... x factors[DIM-1](:,j)
    factors[d] is Ls[d] x N (column major), so the full operator is (prod Ls) x N
    with the last dimension running fastest (same layout as get_polynomials).
    Only the factors are stored (sum_d Ls[d]*N entries); products are computed
    particle by particle (matrix-free).
  */
  template<size_t DIM, typename T>
  class TensorP2M{
  private:
    std::vector<std::vector<T> > factors;
    std::vector<int>             Ls;
    int                          N = 0;

    // s (size prod Ls) <- column j, by successive Kronecker products (in place)
    void expand(int j, T* s) const {
      s[0] = T(1.);
      int n = 1;
      for(size_t k = 0; k < DIM; k++){
	int Lk = Ls[k];
	const T* v = factors[k].data() + j*Lk;
	for(int a = n-1; a >= 0; a--){
	  T sa = s[a];
	  T* sL = s + a*Lk;
	  for(int i = 0; i < Lk; i++){sL[i] = sa * v[i];}
	}
	n *= Lk;
      }
    }

  public:

    TensorP2M(){}

    TensorP2M(T** factors_, int* Ls_, int N_){set(factors_,Ls_,N_);}

    // Copies the factors
    void set(T** factors_, int* Ls_, int N_){
      resize(Ls_,N_);
      for(size_t d = 0; d < DIM; d++){
	factors[d].assign(factors_[d], factors_[d] + Ls[d]*N);}
    }

    // Allocates the factors (uninitialised values)
    void resize(int* Ls_, int N_){
      N = N_;
      Ls.assign(Ls_, Ls_+DIM);
      factors.resize(DIM);
      for(size_t d = 0; d < DIM; d++){factors[d].resize(Ls[d]*N);}
    }

    int rows() const {int r = 1; for(size_t d = 0; d < DIM; d++){r *= Ls[d];} return r;}
    int cols() const {return N;}
    int order (int d) const {return Ls[d];}
    T*  factor(int d)       {return factors[d].data();}

    // Dense (prod Ls) x N matrix, allocated if S == nullptr
    void to_dense(T*& S) const {
      int Ld = rows();
      if(S == nullptr){S = new T[Ld*N];}
      for(int j = 0; j < N; j++){expand(j, S + j*Ld);}
    }

    // C (rows x nrhs) = A B,   B is N x nrhs          (P2M)
    friend void gemm(const TensorP2M<DIM,T>& A, T* B, T* C, int nrhs){
      int Ld = A.rows();
      std::vector<T> s(Ld);
      for(int i = 0; i < Ld*nrhs; i++){C[i] = T(0.);}
      for(int j = 0; j < A.N; j++){
	A.expand(j, s.data());
	for(int c = 0; c < nrhs; c++){
	  T  qj = B[c*A.N + j];
	  T* Cc = C + c*Ld;
	  for(int i = 0; i < Ld; i++){Cc[i] += qj * s[i];}
	}
      }
    }

    // C (N x nrhs) = A^T B,   B is rows x nrhs      (L2P)
    friend void gemTm(const TensorP2M<DIM,T>& A, T* B, T* C, int nrhs){
      int Ld = A.rows();
      std::vector<T> s(Ld);
      for(int j = 0; j < A.N; j++){
	A.expand(j, s.data());
	for(int c = 0; c < nrhs; c++){
	  const T* Bc = B + c*Ld;
	  T res = T(0.);
	  for(int i = 0; i < Ld; i++){res += s[i] * Bc[i];}
	  C[c*A.N + j] = res;
	}
      }
    }

  }; // TensorP2M

  /*
    C = K A  with K = K_0 x ... x K_{DIM-1} (Kron) and A a TensorP2M:
    by the mixed-product property, C is the TensorP2M with factors K_d A_d.
    Requires K.cols(d) == A.order(d). C may be A.
  */
  template<size_t DIM, typename T>
  void gemm(Kron<DIM,T>& K, TensorP2M<DIM,T>& A, TensorP2M<DIM,T>& C){
    int N = A.cols();
    int Ls[DIM];
    std::vector<std::vector<T> > f(DIM);
    for(size_t d = 0; d < DIM; d++){
      Ls[d] = K.rows(d);
      f[d].resize(Ls[d]*N);
      theia::gemm(1.,K.matrix(d),A.factor(d),0.,f[d].data(),K.rows(d),K.cols(d),N);
    }
    T* ptrs[DIM];
    for(size_t d = 0; d < DIM; d++){ptrs[d] = f[d].data();}
    C.set(ptrs,Ls,N);
  }

  /*
    C = K^T A  (factors K_d^T A_d). Requires K.rows(d) == A.order(d). C may be A.
  */
  template<size_t DIM, typename T>
  void gemTm(Kron<DIM,T>& K, TensorP2M<DIM,T>& A, TensorP2M<DIM,T>& C){
    int N = A.cols();
    int Ls[DIM];
    std::vector<std::vector<T> > f(DIM);
    for(size_t d = 0; d < DIM; d++){
      Ls[d] = K.cols(d);
      f[d].resize(Ls[d]*N);
      theia::gemTm(1.,K.matrix(d),A.factor(d),0.,f[d].data(),K.cols(d),K.rows(d),N);
    }
    T* ptrs[DIM];
    for(size_t d = 0; d < DIM; d++){ptrs[d] = f[d].data();}
    C.set(ptrs,Ls,N);
  }

} // THEIA
#endif
