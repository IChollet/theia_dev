#ifndef THEIA_PARSER_EIGENVALUES_HPP
#define THEIA_PARSER_EIGENVALUES_HPP

#include "./blas_parser_consts.hpp"

extern "C"{
  // Eigenvalues of symmetric matrix
  void ssyev_(const char*, const char*, const int*, float*, const int*, float*, float*, const int*, int*);
  void dsyev_(const char*, const char*, const int*, double*, const int*, double*, double*, const int*, int*);
  void cheev_(const char* jobz, const char* uplo, const int* n, std::complex<float>* a, const int* lda, float* w, std::complex<float>* work, const int* lwork, float* rwork, int* info);
    void zheev_(const char* jobz, const char* uplo, const int* n, std::complex<double>* a, const int* lda, double* w, std::complex<double>* work, const int* lwork, double* rwork, int* info);

  // Eigenvalues of symmetric tridiagonal matrix
  void dstevr_(const char* jobz, const char* range, const int* n, double* d, double* e, const double* vl, const double* vu, const int* il, const int* iu, const double* abstol, int* m, double* w, double* z, const int* ldz, int* isuppz, double* work, const int* lwork, int* iwork, const int* liwork, int* info);
  void sstevr_(const char* jobz, const char* range, const int* n, float* d, float* e, const float* vl, const float* vu, const int* il, const int* iu, const float* abstol, int* m, float* w, float* z, const int* ldz, int* isuppz, float* work, const int* lwork, int* iwork, const int* liwork, int* info);

}

namespace theia{

  template<typename FLT> inline void syev(int n, FLT* a, FLT* w){
    std::cout << "Real SYEV undefined for called template" << std::endl; exit(1);}
  template<typename FLT> inline void syev(int n, std::complex<FLT>* a, FLT* w){
    std::cout << "Complex SYEV undefined for called template" << std::endl; exit(1);}
  template<> inline void syev<float>(int n, float* a, float* w){
    char jobz = 'V';
    char uplo = 'U';
    int lda = n;
    int info = 0;
    int lwork = -1;
    float query = 0.0f;
    ssyev_(&jobz, &uplo, &n, a, &lda, w, &query, &lwork, &info);
    lwork = static_cast<int>(query);
    std::vector<float> work(static_cast<std::size_t>(lwork));
    ssyev_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, &info);
    if(info != 0) throw std::runtime_error("LAPACK ssyev failed");
  }
  template<> inline void syev<double>(int n, double* a, double* w){
    char jobz = 'V';
    char uplo = 'U';
    int lda = n;
    int info = 0;
    int lwork = -1;
    double query = 0.0;
    dsyev_(&jobz, &uplo, &n, a, &lda, w, &query, &lwork, &info);
    lwork = static_cast<int>(query);
    std::vector<double> work(static_cast<std::size_t>(lwork));
    dsyev_(&jobz, &uplo, &n, a, &lda, w, work.data(), &lwork, &info);
    if(info != 0) throw std::runtime_error("LAPACK dsyev failed");
  }  
  template<> inline void syev<float>(int n, std::complex<float>* a, float* w){
    char jobz = 'V';
    char uplo = 'U';
    int lda  = n;
    int info = 0;
    int lwork = -1;
    std::complex<float> work_query;
    float rwork_query;
    cheev_(&jobz,&uplo,&n,a,&lda,w,&work_query,&lwork,&rwork_query,&info);
    lwork = static_cast<int>(std::real(work_query));
    std::vector<std::complex<float>> work(lwork);
    std::vector<float> rwork(std::max(1, 3*n - 2));
    cheev_(&jobz,&uplo,&n,a,&lda,w,work.data(),&lwork,rwork.data(),&info);
    if(info != 0) throw std::runtime_error("LAPACK csyev failed");
  }
  template<> inline void syev<double>(int n, std::complex<double>* a, double* w){
    char jobz = 'V';
    char uplo = 'U';
    int lda  = n;
    int info = 0;
    int lwork = -1;
    std::complex<double> work_query;
    double rwork_query;
    zheev_(&jobz,&uplo,&n,a,&lda,w,&work_query,&lwork,&rwork_query,&info);
    lwork = static_cast<int>(std::real(work_query));
    std::vector<std::complex<double>> work(lwork);
    std::vector<double> rwork(std::max(1, 3*n - 2));
    zheev_(&jobz,&uplo,&n,a,&lda,w,work.data(),&lwork,rwork.data(),&info);
    if(info != 0) throw std::runtime_error("LAPACK csyev failed");
  }
} // THEIA

#endif
