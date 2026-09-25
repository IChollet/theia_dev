#ifndef THEIA_BEYLKIN_MONZON_HPP
#define THEIA_BEYLKIN_MONZON_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include "../blas/blas.hpp"
#include "../cubature/cubature.hpp"

namespace theia{
  namespace beylkin_monzon{

    template <typename FLT>
    void gauss_legendre(FLT a, FLT b, int n, std::vector<FLT>& x, std::vector<FLT>& w) {
      auto rule = theia::cubature::gauss_legendre::get<FLT>(n);
      x.assign(n, FLT(0));
      w.assign(n, FLT(0));
      const FLT xm = FLT(0.5) * (b + a);
      const FLT xl = FLT(0.5) * (b - a);
      for (int i = 0; i < n; i++){
	x[i] = xm - xl*rule.x[i][0];
	w[i] = rule.w[i] * (b-a)/2.;
      }
    }

    template <typename FLT> std::vector<std::complex<FLT>>
    kung_hankel_exponents(const std::vector<std::complex<FLT>>& f, FLT h, int M,
			  std::vector<FLT>* singular_values_out = nullptr){

      // Settings
      const int N = static_cast<int>(f.size());
      const int L = N / 2;
      const int Mi = static_cast<int>(M);
      if (Mi<=0 || Mi>=L) throw std::invalid_argument("kung_hankel_exponents: 0 < M < N/2");

      // Hankel matrix from data (column major)
      std::vector<std::complex<FLT>> H0(L*L);
      std::vector<std::complex<FLT>> H1(L*L);
      for(int j = 0; j < L; j++){
	for(int i = 0; i < L; i++){
	  H0[i + j*L] = f[i + j];
	  H1[i + j*L] = f[i + j + 1];
	}
      }

      // SVD of the Hankel matrix
      std::vector<FLT> S;
      std::vector<std::complex<FLT>> U, VT;
      std::vector<std::complex<FLT>> H0_copy = H0; // erased by lapack svd
      theia::svd<FLT>(H0_copy.data(), L, L, U, S, VT);
      if(singular_values_out) *singular_values_out = S;

      // Z = S_M^{-1} * (U_M^H * H1 * V_M)   (MxM matrix)
      // U_M = M first columns of U ; V_M = M first columns of V
      // (V(k,j) = conj(VT(j,k)))
      std::vector<std::complex<FLT>> UH_H1(Mi*L);
      for (int i = 0; i < Mi; i++){
	for (int j = 0; j < L; j++){
	  std::complex<FLT> s(0, 0);
	  for (int k = 0; k < L; ++k)
	    s += std::conj(U[k + i*L]) * H1[k + j*L];
	  UH_H1[i + j*Mi] = s;
	}
      }

      // Z = S_M^{-1}  U_M^{H}  H_1  V_M
      std::vector<std::complex<FLT>> Z(Mi*Mi);
      for(int i = 0; i < Mi; i++){
	for(int jj = 0; jj < Mi; jj++){
	  std::complex<FLT> s(0, 0);
	  for (int k = 0; k < L; k++){
	    s += UH_H1[i + k*Mi] * std::conj(VT[jj + k*L]);
	  }
	  Z[i + jj*Mi] = s/S[i];
	}
      }

      // Get modes
      std::vector<std::complex<FLT>> Z_copy = Z; // erased by eigenvalues
      std::vector<std::complex<FLT>> z;
      theia::eigenvalues<FLT>(Z_copy.data(), Mi, z);
      std::vector<std::complex<FLT>> lambda(Mi);
      for(int i = 0; i < Mi; i++){
	auto zi = z[i];
	if(std::abs(zi) < FLT(1e-300)) zi = std::complex<FLT>(FLT(1e-300), 0);
	lambda[i] = -std::log(zi)/h;
      }

      // Return result
      return lambda;
    }

    // ======================================================================
    // Conversion generique vers/depuis std::complex<FLT>
    // ======================================================================
    template <typename FLT>
    inline std::complex<FLT> to_complex(FLT x){return std::complex<FLT>(x, FLT(0));}
    template <typename FLT>
    inline std::complex<FLT> to_complex(std::complex<FLT> x){return x;}
    template <typename T, typename FLT>
    inline T from_complex(const std::complex<FLT>& v){
      if constexpr (std::is_same_v<T, std::complex<FLT>>){return v;
      }else{return static_cast<T>(v.real());}
    }

    template <typename FLT>
    struct Result {
      std::vector<std::complex<FLT>> coefficients;  // c_i (taille M)
      std::vector<std::complex<FLT>> modes;         // lambda_i (taille M)
      FLT relative_error = FLT(0);
      bool converged = false;
      std::size_t M = 0;
    };

    // Get the Beylkin-Monzon approximation of a 1D positive kernel between Rminus and Rmax
    template <std::size_t DIM, typename FLT, typename T = FLT, typename Kernel>
    Result<FLT> approximation(Kernel kernel, FLT Rminus, FLT Rplus,
			      FLT target_rel_error, std::size_t M_max = 40,
			      std::size_t n_samples = 400, std::size_t n_quad = 300){

      // Raise error if needed
      static_assert(DIM >= 1, "DIM doit etre >= 1");
      static_assert(std::is_floating_point_v<FLT>, "FLT doit etre float ou double");
      static_assert(std::is_same_v<T, FLT> || std::is_same_v<T, std::complex<FLT>>, "T doit etre FLT ou std::complex<FLT>");
      static_assert(std::is_invocable_r_v<T, Kernel, FLT>, "kernel doit avoir la signature T(FLT)");
      if (!(Rminus > FLT(0)) || !(Rplus > Rminus))
	throw std::invalid_argument("bm::fit: il faut 0 < Rminus < Rplus");
      if (n_samples < 8)
	throw std::invalid_argument("bm::fit: n_samples trop petit");

      // Sampling
      const FLT u_m = Rminus * Rminus;
      const FLT u_p = Rplus * Rplus;
      const FLT h = (u_p - u_m) / FLT(n_samples - 1);
      std::vector<std::complex<FLT>> f_samples(n_samples);
      for (std::size_t n = 0; n < n_samples; ++n) {
	FLT u = u_m + h * FLT(n);
	FLT R = std::sqrt(u);
	f_samples[n] = to_complex<FLT>(kernel(R));
      }

      // Gauss-Legendre nodes ponderated with R^{DIM-1}
      std::vector<FLT> Rq, Wq;
      gauss_legendre<FLT>(Rminus, Rplus, n_quad, Rq, Wq);
      std::vector<FLT> sqrt_w(n_quad);
      std::vector<std::complex<FLT>> Kq(n_quad);
      for (std::size_t i = 0; i < n_quad; ++i) {
	FLT w = Wq[i] * std::pow(Rq[i], DIM-1);
	sqrt_w[i] = std::sqrt(w);
	Kq[i] = to_complex<FLT>(kernel(Rq[i]));
      }

      // Independent fine grid to test the error
      const std::size_t n_eval = 2000;
      std::vector<FLT> Reval(n_eval);
      std::vector<std::complex<FLT>> Ktrue(n_eval);
      for (std::size_t i = 0; i < n_eval; ++i) {
	FLT R = Rminus + (Rplus - Rminus) * FLT(i) / FLT(n_eval - 1);
	Reval[i] = R;
	Ktrue[i] = to_complex<FLT>(kernel(R));
      }
      FLT Ktrue_norm2 = FLT(0);
      for (auto& v : Ktrue) Ktrue_norm2 += std::norm(v);
      const FLT Ktrue_norm = std::sqrt(Ktrue_norm2);

      // Settings for the core loop
      Result<FLT> best;
      best.relative_error = std::numeric_limits<FLT>::infinity();
      const std::size_t L = n_samples / 2;
      const std::size_t M_hi = std::min(M_max, L > 1 ? L - 1 : 1);

      // Incremental try to find best rule
      for (std::size_t M = 1; M <= M_hi; ++M) {

	// -> Get modes
	std::vector<std::complex<FLT>> lambda;
	try {
	  lambda = kung_hankel_exponents<FLT>(f_samples, h, M);
	} catch (...) { continue;}

	// -> Get coefficients by solving Vandermond system using least squares
	std::vector<std::complex<FLT>> A(n_quad * M);
	for(std::size_t i = 0; i < M; i++){
	  for(std::size_t q = 0; q < n_quad; q++){
	    A[q + i*n_quad] = sqrt_w[q] * std::exp(-lambda[i] * Rq[q] * Rq[q]);
	  }
	}
	std::vector<std::complex<FLT>> b(std::max(n_quad, M), std::complex<FLT>(0));
	for(std::size_t q = 0; q < n_quad; q++){b[q] = sqrt_w[q] * Kq[q];}
	try {
	  theia::lstsq<FLT>(A.data(), n_quad, M, b.data());
	} catch (...) { continue;}
	std::vector<std::complex<FLT>> c(M);
	for (std::size_t i = 0; i < M; ++i){c[i] = b[i];} // b erased by the LS solution

	// -> Compute L2 error on fine grid
	FLT err_num2 = FLT(0);
	for(std::size_t q = 0; q < n_eval; q++){
	  std::complex<FLT> approx(0, 0);
	  FLT R2 = Reval[q] * Reval[q];
	  for (std::size_t i = 0; i < M; ++i)
	    approx += c[i] * std::exp(-lambda[i] * R2);
	  err_num2 += std::norm(Ktrue[q] - approx);
	}
	FLT rel_error = std::sqrt(err_num2) / Ktrue_norm;

	// -> Check error and decide to continue or not
	if (rel_error < best.relative_error){
	  best.relative_error = rel_error;
	  best.M = M;
	  best.coefficients = c;
	  best.modes = lambda;
	}
	if (best.relative_error <= target_rel_error) break;
      }

      // Return result
      best.converged = (best.relative_error <= target_rel_error);
      return best;
    }

    // ======================================================================
    // Evaluation du modele ajuste (utile pour valider / utiliser le
    // resultat) :  K(x) ~= sum_i c_i * exp(-lambda_i * |x|^2)
    // ======================================================================
    template <std::size_t DIM, typename FLT, typename T>
    T evaluate(const Result<FLT>& res, const std::array<FLT, DIM>& x) {
      FLT r2 = FLT(0);
      for (std::size_t d = 0; d < DIM; ++d) r2 += x[d] * x[d];
      std::complex<FLT> acc(0, 0);
      for (std::size_t i = 0; i < res.M; ++i)
	acc += res.coefficients[i] * std::exp(-res.modes[i] * r2);
      return from_complex<T, FLT>(acc);
    }

    // Evaluation directe en fonction du rayon R (equivalent, plus simple
    // quand on veut juste revalider K(R), sans construire de point x).
    template <typename FLT, typename T>
    T evaluate_radial(const Result<FLT>& res, FLT R) {
      FLT r2 = R * R;
      std::complex<FLT> acc(0, 0);
      for (std::size_t i = 0; i < res.M; ++i)
	acc += res.coefficients[i] * std::exp(-res.modes[i] * r2);
      return from_complex<T, FLT>(acc);
    }
    
  }// BEYLKIN_MONZON
} // THEIA

#endif
