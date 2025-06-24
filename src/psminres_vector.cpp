// Hybrid Parallel sMINRES Method with MPI-Based Domain-Decomposed Vectors and OpenMP-Based Parallelism for Shift Loop and SpMV
#include "mpi_raii.hpp"
#include "psminres_utils.hpp"
#include "psminres_blas.hpp"
#include <iostream>
#include <iomanip>
#include <omp.h>


int main(int argc, char** argv) {
  // MPI を起動
  MPIEnv mpi(argc, argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // 行列の用意
  std::string Aname;
  if (argc < 2) Aname = "../../../GSMINRESpp/data/ELSES_MATRIX_BNZ30_A.mtx";
  else          Aname = argv[1];
  auto A = utils::load_distributed_mm_csr(Aname, rank, size, MPI_COMM_WORLD);
  std::size_t n_global = A.n_global, n_local = A.row_end - A.row_start;
  // 右辺ベクトルの用意
  std::vector<std::complex<double>> b_local(n_local, {1.0, 0.0});
  double bnrm = blas::dznrm2_mpi(n_local, b_local, MPI_COMM_WORLD);
  // シフトの用意
  std::size_t M;
  std::vector<std::complex<double>> sigma;
  utils::set_shift(M, sigma);
  // 解ベクトルの用意
  std::vector<std::vector<std::complex<double>>> x_local(M, std::vector<std::complex<double>>(n_local, {0.0, 0.0}));

  // 変数の用意
  double alpha, beta_prev=0.0, beta_curr=0.0;
  std::vector<std::complex<double>> v_prev(n_local, {0.0,0.0}), v_curr(n_local, {0.0,0.0}), v_next(n_local, {0.0,0.0});
  std::complex<double> T_prev2, T_prev, T_curr, T_next;
  std::vector<std::array<double, 3>>               Gc(M, std::array<double, 3>{0.0, 0.0, 0.0});
  std::vector<std::array<std::complex<double>, 3>> Gs(M, std::array<std::complex<double>, 3>{{{0.0,0.0}, {0.0,0.0}, {0.0,0.0}}});
  std::vector<std::vector<std::complex<double>>> p_prev2(M, std::vector<std::complex<double>>(n_local, {0.0, 0.0}));
  std::vector<std::vector<std::complex<double>>> p_prev(M, std::vector<std::complex<double>>(n_local, {0.0, 0.0}));
  std::vector<std::vector<std::complex<double>>> p_curr(M, std::vector<std::complex<double>>(n_local, {0.0, 0.0}));
  std::vector<std::complex<double>> f(M, {1.0, 0.0});
  std::vector<double> h(M, bnrm);
  std::size_t conv_num = 0, local_conv_num = 0;
  std::vector<std::size_t> is_conv(M, 0);
  utils::Timer timer;

  // 行列ベクトル積の計算のために使用
  std::vector<std::complex<double>> v_global(n_global); 
  std::vector<int> recvcounts(size), displs(size);
  int n_local_int = static_cast<int>(n_local);
  MPI_Allgather(&n_local_int, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);
  displs[0] = 0;
  for (int r = 1; r < size; ++r) displs[r] = displs[r-1] + recvcounts[r-1];

  timer.start();
  blas::zcopy(n_local, b_local, v_curr);
  blas::zdscal(n_local, 1.0/bnrm, v_curr);
  for (std::size_t j = 1; j < 10000; ++j) {
    // 行列ベクトル積を行うために各プロセスから v_curr を v_global に集める
    MPI_Allgatherv(v_curr.data(), n_local_int, MPI_CXX_DOUBLE_COMPLEX,
                   v_global.data(), recvcounts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                   MPI_COMM_WORLD);
    utils::SpMV(A, v_global, v_next);
    alpha = std::real(blas::zdotc_mpi(n_local, v_curr, v_next, MPI_COMM_WORLD));
    blas::zaxpy(n_local, -beta_prev, v_prev, v_next);
    blas::zaxpy(n_local, -alpha,     v_curr, v_next);
    beta_curr = blas::dznrm2_mpi(n_local, v_next, MPI_COMM_WORLD);
    blas::zdscal(n_local, 1.0/beta_curr, v_next);
    local_conv_num = 0;
#pragma omp parallel for private(T_prev2, T_prev, T_curr, T_next) reduction(+:local_conv_num)
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
      blas::zcopy(n_local, p_prev[m], p_prev2[m]);
      blas::zcopy(n_local, p_curr[m], p_prev[m]);
      blas::zcopy(n_local, v_curr,    p_curr[m]);
      blas::zaxpy(n_local, -T_prev2, p_prev2[m], p_curr[m]);
      blas::zaxpy(n_local, -T_prev,  p_prev[m],  p_curr[m]);
      blas::zscal(n_local, 1.0/T_curr, p_curr[m]);
      blas::zaxpy(n_local, bnrm*Gc[m][2]*f[m], p_curr[m], x_local[m]);
      f[m] = -std::conj(Gs[m][2]) * f[m];
      h[m] = std::abs(-std::conj(Gs[m][2])) * h[m];
      if (h[m]/bnrm < 1e-13) {
        local_conv_num++;
        is_conv[m] = j;
      }
      Gc[m][0] = Gc[m][1]; Gc[m][1] = Gc[m][2];
      Gs[m][0] = Gs[m][1]; Gs[m][1] = Gs[m][2];
    }
    beta_prev = beta_curr;
    blas::zcopy(n_local, v_curr, v_prev);
    blas::zcopy(n_local, v_next, v_curr);
    conv_num += local_conv_num;
    if (conv_num >= M) {
      break;
    }
    /*
    if (j % 5 == 1) {
      if (rank == 0) std::cout << j << " ";
      for (std::size_t m = 0; m < M/2; ++m){
        std::vector<std::complex<double>> x_global(n_global), r_local(n_local);
        MPI_Allgatherv(x_local[m].data(), n_local, MPI_CXX_DOUBLE_COMPLEX,
                       x_global.data(), recvcounts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                       MPI_COMM_WORLD);
        utils::SpMV(A, x_global, r_local);
        blas::zaxpy(n_local, sigma[m], x_local[m], r_local);
        blas::zaxpy(n_local, {-1.0, 0.0}, b_local, r_local);
        double tmp_nrm = blas::dznrm2_mpi(n_local, r_local, MPI_COMM_WORLD);
        if (rank == 0)
          std::cout << std::scientific << std::setw(12) << std::setprecision(5)
                    << h[m]/bnrm << " " << tmp_nrm/bnrm;
      }
      if (rank == 0) std::cout << std::endl;
    }
    */
  }
  timer.stop();

  if (rank == 0)
    std::cout << "# Hybrid Parallel sMINRES Method with MPI-Based Domain-Decomposed Vectors and OpenMP-Based Parallelism for Shift Loop and SpMV\n"
              << "# A = " << Aname << "\n"
              << "# status = " << conv_num << "/" << M << ", "
              << "time = " << timer.elapsed_sec()  << " sec"<< std::endl;
  for (std::size_t m = 0; m < M; ++m) {
    std::vector<std::complex<double>> x_global(n_global), r_local(n_local, {0.0, 0.0});
    MPI_Allgatherv(x_local[m].data(), n_local, MPI_CXX_DOUBLE_COMPLEX,
                   x_global.data(), recvcounts.data(), displs.data(), MPI_CXX_DOUBLE_COMPLEX,
                   MPI_COMM_WORLD);
    utils::SpMV(A, x_global, r_local);
    blas::zaxpy(n_local, sigma[m], x_local[m], r_local);
    blas::zaxpy(n_local, {-1.0, 0.0}, b_local, r_local);
    double tmp_nrm = blas::dznrm2_mpi(n_local, r_local, MPI_COMM_WORLD);
    if (rank == 0)
      std::cout << std::right
                << std::setw(2) << m << " "
                << std::fixed << std::setw(10) << std::setprecision(6) << sigma[m].real() << " "
                << std::fixed << std::setw(10) << std::setprecision(6) << sigma[m].imag() << " "
                << std::setw(5) << is_conv[m] << " "
                << std::scientific << std::setw(12) << std::setprecision(5) << h[m] << " "
                << std::scientific << std::setw(12) << std::setprecision(5) << tmp_nrm << "\n";
  }

  return 0;
}
