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
#ifndef THEIA_SYMBOLIC_HPP
#define THEIA_SYMBOLIC_HPP
#include <iostream>
#include <cmath>
#include <complex>
#include <array>

namespace theia{

  template<int DIM,int KRNL,typename T,typename FLT>
  inline T kernel(std::array<FLT,DIM>& x, std::array<FLT,DIM>& y){
    std::cout << "ERROR: kernel<int,int,typename,typename> undefined for arbitrary template!" << std::endl;
    exit(1);
  }

  template<int DIM, int KRNL, typename FLT, typename T>
  inline void get_symbolic_matrix(std::array<FLT,DIM>* px, std::array<FLT,DIM>* py, 
				  int Nx, int Ny, T*& A){
    for(int j = 0; j < Ny; j++){
      for(int i = 0; i < Nx; i++){
	A[j*Nx+i] = kernel<DIM,KRNL,T,FLT>(px[i],py[j]);
      }
    }
  }

  
  template<int DIM, typename FLT, typename T, class KRNL>
  inline void get_symbolic_matrix(std::array<FLT,DIM>* px, std::array<FLT,DIM>* py, 
				  int Nx, int Ny, T*& A, KRNL* K){
    (*K)(px, Nx, py, Ny, A);
  }

}// THEIA
#endif
