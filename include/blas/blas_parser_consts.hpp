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
#include <complex>
#include <vector>
#include <iostream>
#ifndef BLAS_PARSER_CONSTS_THEIA
#define BLAS_PARSER_CONSTS_THEIA
#define BLAS_S float
#define BLAS_D double
#define BLAS_C std::complex<float>
#define BLAS_Z std::complex<double>
constexpr BLAS_S S_ZERO     =  0.0;
constexpr BLAS_S S_ONE      =  1.0;
constexpr BLAS_S S_MONE     = -1.0;
constexpr BLAS_D D_ZERO     =  0.0;
constexpr BLAS_D D_ONE      =  1.0;
constexpr BLAS_D D_MONE     = -1.0;
constexpr BLAS_C C_ZERO     =  0.0;
constexpr BLAS_C C_ONE      =  1.0;
constexpr BLAS_C C_MONE     = -1.0;
constexpr BLAS_Z Z_ZERO     =  0.0;
constexpr BLAS_Z Z_ONE      =  1.0;
constexpr BLAS_Z Z_MONE     = -1.0;
constexpr int IN_ONE     =  1  ;
constexpr const char* charN = "N" ;
constexpr const char* charT = "T" ;
constexpr const char* charC = "C" ;
constexpr const char* charS = "S" ;
constexpr const char* charA = "A" ;

#endif
