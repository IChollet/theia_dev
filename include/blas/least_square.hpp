#ifndef THEIA_BLAS_LEAST_SQUARE_HPP
#define THEIA_BLAS_LEAST_SQUARE_HPP
#include "./blas_parser_consts.hpp"

extern "C" {
  
// Moindres carres complexes (SVD-based, robuste au rang deficient)
void cgelss_(const int* m, const int* n,
             const int* nrhs, std::complex<float>* a,
             const int* lda, std::complex<float>* b,
             const int* ldb, float* s, const float* rcond,
             int* rank, std::complex<float>* work,
             const int* lwork, float* rwork, int* info);
void zgelss_(const int* m, const int* n,
             const int* nrhs, std::complex<double>* a,
             const int* lda, std::complex<double>* b,
             const int* ldb, double* s, const double* rcond,
             int* rank, std::complex<double>* work,
             const int* lwork, double* rwork, int* info);
} // extern "C"

namespace theia{

  // Compute the solution of the complex least square problem min_x ||Ax-b||
  // WARNING: this routine erases b to store the result
  template <typename FLT>
  void lstsq(std::complex<FLT>* A, int m, int n, std::complex<FLT>* b) {
    int lda = m, ldb = std::max(m, n), nrhs = 1, info = 0, rank = 0, lwork = -1;
    int minmn = std::min(m, n);
    std::vector<FLT> S(static_cast<std::size_t>(minmn));
    std::vector<FLT> rwork(static_cast<std::size_t>(std::max(int(1), 5 * minmn)));
    FLT rcond = FLT(-1);
    std::complex<FLT> work_query{};
    auto call = [&](std::complex<FLT>* work_ptr, int lwork_val) {
      if constexpr (std::is_same_v<FLT, float>){
	cgelss_(&m, &n, &nrhs, A, &lda, b, &ldb, S.data(),
		&rcond, &rank, work_ptr, &lwork_val,
		rwork.data(), &info);
      }else{
	zgelss_(&m, &n, &nrhs, A, &lda, b, &ldb, S.data(),
		&rcond, &rank, work_ptr, &lwork_val,
		rwork.data(), &info);
      }
    };
    call(&work_query, -1);
    lwork = static_cast<int>(work_query.real());
    std::vector<std::complex<FLT>> work(static_cast<std::size_t>(std::max(int(1), lwork)));
    call(work.data(), lwork);
    if (info != 0)
      throw std::runtime_error("lstsq: ?gelss info=" +
			       std::to_string(static_cast<long long>(info)));
  }
  
} // THEIA


#endif
