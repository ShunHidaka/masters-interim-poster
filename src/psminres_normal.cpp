// No Parallel sMINRES Method (OpenMP parallelism SpMV)
#include "psminres_utils.hpp"
#include "psminres_blas.hpp"
#include <iostream>
#include <iomanip>


int main(int argc, char** argv) {
  // 行列の用意
  std::string Aname;
  if (argc < 2) Aname = "../../../GSMINRESpp/data/ELSES_MATRIX_BNZ30_A.mtx";
  else          Aname = argv[1];
  auto A = utils::load_mm_csr(Aname);
  std::size_t N = A.row_end - A.row_start;
  // 右辺ベクトルの用意
  std::vector<std::complex<double>> b(N, {1.0, 0.0});
  double bnrm = blas::dznrm2(N, b);
  // シフトの用意
  std::size_t M;
  std::vector<std::complex<double>> sigma;
  utils::set_shift(M, sigma);
  // 解ベクトルの用意
  std::vector<std::vector<std::complex<double>>> x(M, std::vector<std::complex<double>>(N, {0.0, 0.0}));

  // 変数の用意
  double alpha, beta_prev=0.0, beta_curr=0.0;
  std::vector<std::complex<double>> v_prev(N, {0.0,0.0}), v_curr(N, {0.0,0.0}), v_next(N, {0.0,0.0});
  std::complex<double> T_prev2, T_prev, T_curr, T_next;
  std::vector<std::array<double, 3>>               Gc(M, std::array<double, 3>{0.0, 0.0, 0.0});
  std::vector<std::array<std::complex<double>, 3>> Gs(M, std::array<std::complex<double>, 3>{{{0.0,0.0}, {0.0,0.0}, {0.0,0.0}}});
  std::vector<std::vector<std::complex<double>>> p_prev2(M, std::vector<std::complex<double>>(N, {0.0, 0.0}));
  std::vector<std::vector<std::complex<double>>> p_prev(M, std::vector<std::complex<double>>(N, {0.0, 0.0}));
  std::vector<std::vector<std::complex<double>>> p_curr(M, std::vector<std::complex<double>>(N, {0.0, 0.0}));
  std::vector<std::complex<double>> f(M, {1.0, 0.0});
  std::vector<double> h(M, bnrm);
  std::size_t conv_num = 0;
  std::vector<std::size_t> is_conv(M, 0);
  utils::Timer timer;

  timer.start();
  blas::zcopy(N, b, v_curr);
  blas::zdscal(N, 1.0/bnrm, v_curr);
  for (std::size_t j = 1; j < 10000; ++j) {
    utils::SpMV(A, v_curr, v_next);
    alpha = std::real(blas::zdotc(N, v_curr, v_next));
    blas::zaxpy(N, -beta_prev, v_prev, v_next);
    blas::zaxpy(N, -alpha,     v_curr, v_next);
    beta_curr = blas::dznrm2(N, v_next);
    blas::zdscal(N, 1.0/beta_curr, v_next);
    for (std::size_t m = 0; m < M; ++m) {
      if (is_conv[m] != 0){
        continue;
      }
      T_prev2 = 0.0;
      T_prev = beta_prev; T_curr = alpha + sigma[m]; T_next = beta_curr;
      if (j >= 3) {
        blas::zrot(T_prev2, T_prev, Gc[m][0], Gs[m][0]);
      }
      if (j >= 2) {
        blas::zrot(T_prev,  T_curr, Gc[m][1], Gs[m][1]);
      }
      blas::zrotg(T_curr, T_next, Gc[m][2], Gs[m][2]);
      blas::zcopy(N, p_prev[m], p_prev2[m]);
      blas::zcopy(N, p_curr[m], p_prev[m]);
      blas::zcopy(N, v_curr,    p_curr[m]);
      blas::zaxpy(N, -T_prev2, p_prev2[m], p_curr[m]);
      blas::zaxpy(N, -T_prev,  p_prev[m],  p_curr[m]);
      blas::zscal(N, 1.0/T_curr, p_curr[m]);
      blas::zaxpy(N, bnrm*Gc[m][2]*f[m], p_curr[m], x[m]);
      f[m] = -std::conj(Gs[m][2]) * f[m];
      h[m] = std::abs(-std::conj(Gs[m][2])) * h[m];
      if (h[m]/bnrm < 1e-13) {
        conv_num++;
        is_conv[m] = j;
      }
      Gc[m][0] = Gc[m][1]; Gc[m][1] = Gc[m][2];
      Gs[m][0] = Gs[m][1]; Gs[m][1] = Gs[m][2];
    }
    beta_prev = beta_curr;
    blas::zcopy(N, v_curr, v_prev);
    blas::zcopy(N, v_next, v_curr);
    if (conv_num >= M) {
      break;
    }
    /*
    if (j % 5 == 1) {
      std::cout << j << " ";
      for (std::size_t m = 0; m < M/2; ++m){
        std::vector<std::complex<double>> r_local(N);
        utils::SpMV(A, x[m], r_local);
        blas::zaxpy(N, sigma[m], x[m], r_local);
        blas::zaxpy(N, {-1.0, 0.0}, b, r_local);
        double tmp_nrm = blas::dznrm2(N, r_local);
        std::cout << std::scientific
                  << h[m]/bnrm << " " << tmp_nrm/bnrm << ", ";
      }
      std::cout << std::endl;
    }
    */
  }
  timer.stop();

  std::cout << "# No parallel sMINRES Method (OpenMP paralleism SpMV)\n"
            << "# A = " << Aname << "\n"
            << "# status = " << conv_num << "/" << M << ", "
            << "time = " << timer.elapsed_sec()  << " sec"<< std::endl;
  for (std::size_t m = 0; m < M; ++m) {
    std::vector<std::complex<double>> r_local(N, {0.0, 0.0});
    utils::SpMV(A, x[m], r_local);
    blas::zaxpy(N, sigma[m], x[m], r_local);
    blas::zaxpy(N, {-1.0, 0.0}, b, r_local);
    double tmp_nrm = blas::dznrm2(N, r_local);
    std::cout << std::right
              << std::setw(2) << m << " "
              << std::fixed << std::setw(10) << std::setprecision(6) << sigma[m].real() << " "
              << std::fixed << std::setw(10) << std::setprecision(6) << sigma[m].imag() << " "
              << std::setw(5) << is_conv[m] << " "
              << std::scientific << std::setw(12) << std::setprecision(5) << h[m] << " "
              << std::scientific << std::setw(12) << std::setprecision(5) << tmp_nrm << std::endl;
  }

  return 0;
}
