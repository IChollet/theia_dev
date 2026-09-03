#ifndef THEIA_PARSER_GETRF_HPP
#define THEIA_PARSER_GETRF_HPP

#include "./blas_parser_consts.hpp"

extern "C"{
  // Inverse of a general matrix
  void sgetrf_(const int*, const int*, const BLAS_S*, int*, const int*, int*);
  void sgetri_(const int*, const BLAS_S*, const int*, const int*, const BLAS_S*, const int*, const int*);
  void dgetrf_(const int*, const int*, const BLAS_D*, int*, const int*, int*);
  void dgetri_(const int*, const BLAS_D*, const int*, const int*, const BLAS_D*, const int*, const int*);
  void cgetrf_(const int*, const int*, const BLAS_C*, const int*, int*, int*);
  void cgetri_(const int*, const BLAS_C*, const int*, const int*, const BLAS_C*, const int*, const int*);
  void zgetrf_(const int*, const int*, const BLAS_Z*, const int*, int*, int*);
  void zgetri_(const int*, const BLAS_Z*, const int*, const int*, const BLAS_Z*, const int*, const int*);
}

namespace theia{    
  void getrf(int m, int n, float* A, int lda, int* ipiv){
    int info;
    sgetrf_(&m, &n, A, &lda, ipiv, &info);
    if(info < 0){std::cout << "sgetrf: invalid argument"  << std::endl;}
    if(info > 0){std::cout << "sgetrf: singular matrix " << std::endl;}
  }
  void getri(int n, float* A, int lda, int* ipiv){
    int info;
    int lwork = -1;
    float work_query;
    sgetri_(&n, A, &lda, ipiv, &work_query, &lwork, &info);
    if (info != 0){std::cout << "sgetri: erreur workspace query" << std::endl;}
    lwork = (int)work_query;
    float* work = new float[lwork];
    sgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    delete[] work;
    if(info < 0){std::cout << "sgetri: invalid argument" << std::endl;}
    if(info > 0){std::cout << "sgetri: singular matrix " << std::endl;
    }
  }
  void getrf(int m, int n, double* A, int lda, int* ipiv){
    int info;
    dgetrf_(&m, &n, A, &lda, ipiv, &info);
    if(info < 0){std::cout << "sgetrf: invalid argument"  << std::endl;}
    if(info > 0){std::cout << "sgetrf: singular matrix " << std::endl;}
  }
  void getri(int n, double* A, int lda, int* ipiv){
    int info;
    int lwork = -1;
    double work_query;
    dgetri_(&n, A, &lda, ipiv, &work_query, &lwork, &info);
    if (info != 0){std::cout << "sgetri: erreur workspace query" << std::endl;}
    lwork = (int)work_query;
    double* work = new double[lwork];
    dgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    delete[] work;
    if(info < 0){std::cout << "sgetri: invalid argument" << std::endl;}
    if(info > 0){std::cout << "sgetri: singular matrix " << std::endl;
    }
  }
  void getrf(int m, int n, std::complex<float>* A, int lda, int* ipiv){
    int info;
    cgetrf_(&m, &n, A, &lda, ipiv, &info);
    if(info < 0){std::cout << "sgetrf: invalid argument"  << std::endl;}
    if(info > 0){std::cout << "sgetrf: singular matrix " << std::endl;}
  }
  void getri(int n, std::complex<float>* A, int lda, int* ipiv){
    int info;
    int lwork = -1;
    std::complex<float> work_query;
    cgetri_(&n, A, &lda, ipiv, &work_query, &lwork, &info);
    if (info != 0){std::cout << "sgetri: erreur workspace query" << std::endl;}
    lwork = int(work_query.real());
    std::complex<float>* work = new std::complex<float>[lwork];
    cgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    delete[] work;
    if(info < 0){std::cout << "sgetri: invalid argument" << std::endl;}
    if(info > 0){std::cout << "sgetri: singular matrix " << std::endl;
    }
  }
  void getrf(int m, int n, std::complex<double>* A, int lda, int* ipiv){
    int info;
    zgetrf_(&m, &n, A, &lda, ipiv, &info);
    if(info < 0){std::cout << "sgetrf: invalid argument"  << std::endl;}
    if(info > 0){std::cout << "sgetrf: singular matrix " << std::endl;}
  }
  void getri(int n, std::complex<double>* A, int lda, int* ipiv){
    int info;
    int lwork = -1;
    std::complex<double> work_query;
    zgetri_(&n, A, &lda, ipiv, &work_query, &lwork, &info);
    if (info != 0){std::cout << "sgetri: erreur workspace query" << std::endl;}
    lwork = int(work_query.real());
    std::complex<double>* work = new std::complex<double>[lwork];
    zgetri_(&n, A, &lda, ipiv, work, &lwork, &info);
    delete[] work;
    if(info < 0){std::cout << "sgetri: invalid argument" << std::endl;}
    if(info > 0){std::cout << "sgetri: singular matrix " << std::endl;
    }
  }
} // THEIA

#endif
