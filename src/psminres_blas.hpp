#ifndef PSMINRES_BLAS_HPP
#define PSMINRES_BLAS_HPP

#include <complex>
#include <vector>
#include <cstddef>
#include <cstdlib>
#include <mpi.h>

extern "C" {
  void zdscal_(const int* n, const double* alpha,
               std::complex<double>* x, const int* incx);
  void zscal_(const int* n, const std::complex<double>* alpha,
              std::complex<double>* x, const int* incx);
  void zaxpy_(const int* n, const std::complex<double>* alpha,
              const std::complex<double>* x, const int* incx,
              std::complex<double>*       y, const int* incy);
  void zcopy_(const int* n,
              const std::complex<double>* x, const int* incx,
              std::complex<double>* y, const int* incy);
  std::complex<double> zdotc_(const int* n,
                              const std::complex<double>* x, const int* incx,
                              const std::complex<double>* y, const int* incy);
  double dznrm2_(const int* n,
                 const std::complex<double>* x, const int* incx);
  void zrotg_(std::complex<double> *a, std::complex<double> *b,
              double *c, std::complex<double> *s);
  void zrot_(const int *n,
             std::complex<double> *x, const int *incx,
             std::complex<double> *y, const int *incy,
             const double *c, const std::complex<double> *s);
}

namespace blas {

  inline void zdscal(std::size_t n, double alpha,
                     std::vector<std::complex<double>>& x) {
    int nn = static_cast<int>(n);
    int ix = 1;
    zdscal_(&nn, &alpha, x.data(), &ix);
  }

  inline void zscal(std::size_t n, std::complex<double> alpha,
                    std::vector<std::complex<double>>& x) {
    int nn = static_cast<int>(n);
    int ix = 1;
    zscal_(&nn, &alpha, x.data(), &ix);
  }

  inline void zaxpy(std::size_t n, std::complex<double> alpha,
                    const std::vector<std::complex<double>>& x,
                    std::vector<std::complex<double>>& y) {
    int nn = static_cast<int>(n);
    int ix = 1, iy = 1;
    zaxpy_(&nn, &alpha, x.data(), &ix, y.data(), &iy);
  }

  inline void zcopy(std::size_t n,
                    const std::vector<std::complex<double>>& x,
                    std::vector<std::complex<double>>& y) {
    int nn = static_cast<int>(n);
    int ix = 1, iy = 1;
    zcopy_(&nn, x.data(), &ix, y.data(), &iy);
  }

  inline std::complex<double> zdotc(std::size_t n,
                                    const std::vector<std::complex<double>>& x,
                                    const std::vector<std::complex<double>>& y) {
    int nn = static_cast<int>(n);
    int ix = 1, iy = 1;
    return zdotc_(&nn, x.data(), &ix, y.data(), &iy);
  }

  inline std::complex<double> zdotc_mpi(std::size_t n,
                                        const std::vector<std::complex<double>>& x,
                                        const std::vector<std::complex<double>>& y,
                                        MPI_Comm comm) {
    std::complex<double> local = zdotc(n, x, y);
    std::complex<double> global(0.0, 0.0);
    MPI_Allreduce(&local, &global, 1, MPI_CXX_DOUBLE_COMPLEX, MPI_SUM, comm);
    return global;
  }

  inline double dznrm2(std::size_t n,
                       const std::vector<std::complex<double>>& x) {
    int nn = static_cast<int>(n);
    int ix = 1;
    return dznrm2_(&nn, x.data(), &ix);
  }

  inline double dznrm2_mpi(std::size_t n,
                           const std::vector<std::complex<double>>& x,
                           MPI_Comm comm) {
    return std::sqrt(std::real(zdotc_mpi(n, x, x, comm)));
  }

  inline void zrotg(std::complex<double>& a, std::complex<double>& b,
                    double& c, std::complex<double>& s) {
    zrotg_(&a, &b, &c, &s);
  }

  inline void zrot(std::complex<double>& x, std::complex<double>& y,
                   double c, std::complex<double> s) {
    int n = 1;
    int ix = 1, iy = 1;
    zrot_(&n, &x, &ix, &y, &iy, &c, &s);
  }
}

#endif // PSMINRES_BLAS_HPP

/* TODO
 * 
 * blasint なるものがOpenBLASなどではある
 * 大規模問題では int の範囲を超えることがあるので
 * int => long にするためなどで使われる
 * 標準ではないのでとりあえず int のままでいく
 */
