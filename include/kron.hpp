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
#ifndef THEIA_KRON_HPP
#define THEIA_KRON_HPP
#include <vector>
#include <algorithm>
#include "./matrices.hpp"
#include "./nkmm.hpp"

namespace theia{

  template<int DIM>
  inline int swap_multi_idx(int i, int* Ns, int* I){
    int tmp = i;
    for(int d = 0; d < DIM; d++){
      I[DIM-1-d] = (tmp%Ns[DIM-1-d]);
      tmp       /= Ns[DIM-1-d];
    }
    tmp     = 0;
    int dec = 1;
    for(int d = 0; d < DIM; d++){
      tmp += I [((DIM-d)%DIM)] * dec;
      dec *= Ns[((DIM-d)%DIM)];
    }
    return tmp;
  }

  template<int DIM>
  inline void ite_get_permutation(int* I, int* Ns, int k, int* permutation){
    int NN = 1;
    for(int d = 0; d < DIM; d++){NN *= Ns[d];}
    for(int i = 0; i < NN; i++){
      permutation[i] = swap_multi_idx<DIM>(i,Ns,I);
    }
  }

  template<int DIM>
  inline void get_permutations(int* Ms, int* Ns, int**& permutations){
    int prod_of_sizes = 1;
    int*  tmp_Ns       = new int [DIM];
    int*  tmp_Ns_in    = new int [DIM];
    int*  I            = new int [DIM];
    permutations = new int*[DIM];
    for(int d = 0; d < DIM; d++){tmp_Ns[d] = Ns[d];}
    for(int d = 0; d < DIM; d++){prod_of_sizes *= tmp_Ns[d];}
    for(int d = 0; d < DIM; d++){
      tmp_Ns[(DIM-1+d)%DIM] = Ms[(DIM-1+d)%DIM];
      prod_of_sizes  /= Ns[(DIM-1+d)%DIM];
      prod_of_sizes  *= Ms[(DIM-1+d)%DIM];
      permutations[d] = new int[prod_of_sizes];
      for(int k = 0; k < DIM; k++){
      	tmp_Ns_in[k] = tmp_Ns[((DIM+k+d)%DIM)];
      }
      ite_get_permutation<DIM>(I, tmp_Ns_in, 0, permutations[d]);
    }
    delete [] tmp_Ns;
    delete [] tmp_Ns_in;
    delete [] I;
  }
  
  /*
    Kronecker product of DIM matrices (matrices[d] is Ms[d] x Ns[d], column major).
    Matrices and sizes are copied: the input arrays may be freed after set().
  */
  template<size_t DIM, typename T>
  class Kron{
  private:
    std::vector<std::vector<T> >   matrices;
    std::vector<int>               Ms;
    std::vector<int>               Ns;
    std::vector<std::vector<int> > permutations;
    std::vector<std::vector<int> > permutations_transpose;

    static void compute_permutations(std::vector<int>& Ms_, std::vector<int>& Ns_,
				     std::vector<std::vector<int> >& perms){
      int** p;
      get_permutations<DIM>(Ms_.data(),Ns_.data(),p);
      int prod_of_sizes = 1;
      for(size_t d = 0; d < DIM; d++){prod_of_sizes *= Ns_[d];}
      perms.resize(DIM);
      for(size_t d = 0; d < DIM; d++){
	prod_of_sizes /= Ns_[(DIM-1+d)%DIM];
	prod_of_sizes *= Ms_[(DIM-1+d)%DIM];
	perms[d].assign(p[d], p[d]+prod_of_sizes);
	delete [] p[d];
      }
      delete [] p;
    }

    // Shared kernel of gemm / gemTm: TRANS selects A_d or A_d^T
    template<bool TRANS>
    static void apply(std::vector<std::vector<T> >& mats,
		      std::vector<int>& Ms_, std::vector<int>& Ns_,
		      std::vector<std::vector<int> >& perms,
		      T* B, T* C, int nrhs){
      // Ms_/Ns_ here are the sizes of op(A_d) (rows/cols)
      if(perms.empty()){compute_permutations(Ms_,Ns_,perms);}
      int prod_of_sizes = 1;
      int max_prod_of_sizes = 1;
      for(size_t d = 0; d < DIM; d++){
	prod_of_sizes     *= Ns_[d];
	max_prod_of_sizes *= std::max(Ns_[d],Ms_[d]);
      }
      std::vector<T> tmp0(size_t(nrhs)*max_prod_of_sizes);
      std::vector<T> tmp1(size_t(nrhs)*max_prod_of_sizes);
      for(int i = 0; i < prod_of_sizes*nrhs; i++){
	tmp0[i] = B[i];
      }
      for(size_t d = 0; d < DIM; d++){
	int dd = (DIM+d-1)%DIM;
	prod_of_sizes /= Ns_[dd];
	if(TRANS){
	  theia::gemTm(1.,mats[dd].data(),tmp0.data(),
		       0.,tmp1.data(),
		       Ms_[dd],Ns_[dd],prod_of_sizes*nrhs);
	}else{
	  theia::gemm (1.,mats[dd].data(),tmp0.data(),
		       0.,tmp1.data(),
		       Ms_[dd],Ns_[dd],prod_of_sizes*nrhs);
	}
	prod_of_sizes *= Ms_[dd];
	int* perm = perms[d].data();
	for(int r = 0; r < nrhs; r++){
	  T* out = tmp0.data() + r*prod_of_sizes;
	  T* in  = tmp1.data() + r*prod_of_sizes;
	  for(int i = 0; i < prod_of_sizes; i++){
	    out[perm[i]] = in[i];
	  }
	}
      }
      for(int i = 0; i < prod_of_sizes*nrhs; i++){
	C[i] = tmp0[i];
      }
    }
    
  public :

    Kron(){};
    
    Kron(T** matrices_, int* Ms_, int* Ns_){set(matrices_,Ms_,Ns_);}

    void set(T** matrices_, int* Ms_, int* Ns_){
      Ms.assign(Ms_, Ms_+DIM);
      Ns.assign(Ns_, Ns_+DIM);
      matrices.resize(DIM);
      for(size_t d = 0; d < DIM; d++){
	matrices[d].assign(matrices_[d], matrices_[d] + Ms[d]*Ns[d]);}
      permutations.clear();
      permutations_transpose.clear();
    }
        
    void prcmp(){compute_permutations(Ms,Ns,permutations);}

    // Read access to the d-th factor (Ms[d] x Ns[d], column major)
    int rows  (int d) const {return Ms[d];}
    int cols  (int d) const {return Ns[d];}
    T*  matrix(int d)       {return matrices[d].data();}
    
    friend void gemm(Kron<DIM,T>& A, T* B, T* C, int nrhs){
      apply<false>(A.matrices,A.Ms,A.Ns,A.permutations,B,C,nrhs);
    }

    friend void gemTm(Kron<DIM,T>& A, T* B, T* C, int nrhs){
      apply<true>(A.matrices,A.Ns,A.Ms,A.permutations_transpose,B,C,nrhs);
    }
        
  }; // Kron


} // Theia
#endif
