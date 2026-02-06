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
#ifndef THEIA_POLYNOMIALS__HPP
#define THEIA_POLYNOMIALS__HPP
#include <iostream>
#include <cmath>
#include <array>
#include "./matrices.hpp"
#include "./symbolic.hpp"
#include "./kron.hpp"

namespace theia{
    
  inline int myintpow(int a, int b){int res = 1;for(int i = 0; i < b; i++){res *= a;}return res;}

  // Get node
  template<int ITYPE> inline double get_node(int k, int L){
    std::cout << "Error: get_node undefined for general purpose (see general_inrp.hpp)"
	      << std::endl; return 0.;}
  template<> inline double get_node<0>(int k, int L){
    return std::cos((double)(2*k+1)/(double)(2*L)*M_PI);}
  template<> inline double get_node<1>(int k, int L){
    return -1.+2.*double(k)/double(L-1);}
  template<int ITYPE> inline double get_node(int k, int L,
					     const double& a, const double& b){
    std::cout << "Error: get_node undefined for general purpose (see general_inrp.hpp)"
	      << std::endl; return 0.;}
  template<> inline double get_node<0>(int k, int L, const double& a, const double& b){
    return a + (b-a)*(0.5+std::cos((double)(2*k+1)/(double)(2*L)*M_PI)/2.);}
  template<> inline double get_node<1>(int k, int L, const double& a, const double& b){
    return a + (b-a)*(double(k)/double(L-1));}

  // Chebyshev utils
  inline double T(double x, int k){return cos((double)(k)*acos(x));}  
  inline void getTr(int k, int L, double *Tr){
    double r = get_node<0>(k,L);
    for(int i = 0; i < L; i++){Tr[i] = T(r,i);}}
  inline void getTs(double u, int L, double *Tu){
    for(int i = 0; i < L; i++){Tu[i] = T(u,i);}}

  // Lagrange polynomials (0 Chebyshev, 1 equispaced)
  template<int ITYPE>
  inline double C1D(const double& x, int k, int L){
    std::cout << "Error: C1D undefined for general purpose (see general_intrp.hpp)"
	      << std::endl;
    return 0.;
  }
  template<> inline double C1D<0>(const double& x, int k, int L){
    double* Tx = new double[L]; getTs(x,L,Tx);
    double* Tr = new double[L]; getTr(k,L,Tr);
    double res = 1.;
    for(int j = 1; j < L; j++){
      res += 2. * Tx[j] * Tr[j];}
    return res/(double)(L);
  }
  template<> inline double C1D<1>(const double& x, int k, int L){
    double res = 1.;
    double pi  = -1.+2.*((double)( k)/(double)(L-1));
    for(int ii=0; ii<k; ii++){
      double pii = -1.+2.*((double)(ii)/(double)(L-1));
      res *= (x-pii)/(pi-pii);}
    for(int ii=k+1; ii<L; ii++){
      double pii = -1.+2.*((double)(ii)/(double)(L-1));
      res *= (x-pii)/(pi-pii);}
    return res;
  }
  
} // THEIA

#endif
