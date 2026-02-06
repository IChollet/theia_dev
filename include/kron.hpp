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
#include "./matrices.hpp"

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
      int* I = new int[DIM];
      ite_get_permutation<DIM>(I, tmp_Ns_in, 0, permutations[d]);
    }
  }
  
  template<int DIM, typename T>
  class Kron{
  private:
    T**      matrices;
    int*     Ms;
    int*     Ns;
    int**    permutations;
  public :
   
    Kron(T** matrices_, int* Ms_, int* Ns_){
      matrices = matrices_; Ms = Ms_; Ns = Ns_;}
        
    void prcmp(){get_permutations<DIM>(Ms,Ns,permutations);}
    
    friend void gemm(Kron<DIM,T>& A, T* B, T* C, int nrhs){
      T* tmp0 = new T[nrhs];
      T* tmp1 = new T[nrhs];
      int prod_of_sizes = 1;
      for(int d = 0; d < DIM; d++){prod_of_sizes *= A.Ns[d];}
      for(int i = 0; i < prod_of_sizes; i++){tmp0[i] = B[i];}
      for(int d = 0; d < DIM; d++){
	prod_of_sizes /= A.Ns[(DIM+d-1)%DIM];
	theia::gemm(1.,A.matrices[(DIM+d-1)%DIM],tmp0,
	     0.,tmp1,
	     A.Ms[(DIM+d-1)%DIM],A.Ns[(DIM+d-1)%DIM],prod_of_sizes*nrhs);
	prod_of_sizes *= A.Ms[(DIM+d-1)%DIM];
	int* perm = A.permutations[d];
	for(int r = 0; r < nrhs; r++){
	  for(int i = 0; i < prod_of_sizes; i++){
	    (tmp0+r*prod_of_sizes)[perm[i]] = tmp1[i];
	  }
	}
      }
      for(int i = 0; i < prod_of_sizes; i++){
	C[i] = tmp0[i];
      }
    }
    
  }; // Kron


} // Theia
#endif
