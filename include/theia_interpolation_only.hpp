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
#ifndef THEIA_INTERPOLATION_ONLY_HEADER
#define THEIA_INTERPOLATION_ONLY_HEADER

/*
  Entry point restricted to the FMM by interpolation (polynomial interpolation,
  low-rank compression, P2M/M2M/L2L/L2P operators, tensor forms).
  Unlike theia.hpp, this header and everything it includes only require C++14,
  so it can be used from C++14 codes. Linking requires BLAS/LAPACK.
*/
#if defined(_MSVC_LANG) ? (_MSVC_LANG < 201402L) : (__cplusplus < 201402L)
#error "theia_interpolation_only.hpp requires C++14 or later"
#endif

#include "general_intrp.hpp"
#include "operators/fmm_interpolation.hpp"
#include "operators/fmm_tensor_interpolation.hpp"

#endif
