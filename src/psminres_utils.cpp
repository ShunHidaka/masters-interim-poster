#include "psminres_utils.hpp"
#include <algorithm>
#include <fstream>
#include <omp.h>

namespace utils {

  void set_shift(std::size_t& M, std::vector<std::complex<double>>& shift) {
    M = 10;
    shift.resize(M);
    for (std::size_t i = 0; i < M; ++i){
      shift[i] = std::polar(0.01, 2*std::acos(-1)*(i+0.5)/M);
    }
  }

  void load_mm_coo(const std::string& filename,
                   std::size_t& matrix_size,
                   std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>>& entries) {
    // ファイルの展開
    std::ifstream fin(filename);
    if (!fin) {
      throw std::runtime_error("Failed to open file: " + filename);
    }
    // ヘッダ行の読み込み
    std::string line;
    bool is_complex   = false;
    bool is_symmetric = false;
    bool is_hermitian = false;
    if (!std::getline(fin, line)) {
      throw std::runtime_error("Matrix Market header is missing");
    }
    if (line.find("%%MatrixMarket matrix coordinate") != std::string::npos) {
      if(line.find("complex")   != std::string::npos) is_complex   = true;
      if(line.find("symmetric") != std::string::npos) is_symmetric = true;
      if(line.find("hermitian") != std::string::npos) is_hermitian = true;
    } else {
      throw std::runtime_error("Unsupported Matrix Market format");
    }
    if (!((!is_complex && is_symmetric) || (is_complex && is_hermitian))) {
      throw std::runtime_error("Only real-symmetric or complex-hermitian matrix is supported");
    }
    // コメント行のスキップ
    while (std::getline(fin, line)) {
      if (!line.empty() && line[0] != '%') break;
    }
    // 行列の読み込み
    std::istringstream iss(line);
    std::size_t nrows, ncols, nnz = 0;
    if (!(iss >> nrows >> ncols >> nnz)) {
      throw std::runtime_error("Failed to read matrix size line");
    }
    if (nrows != ncols) {
      throw std::runtime_error("Symmetric or Hermitian matrix must be square (nrows != ncols)");
    }
    matrix_size = nrows;
    entries.clear();
    entries.reserve(2*nnz);
    std::size_t i, j;
    double re = 0.0, im = 0.0;
    for (std::size_t k=0; k<nnz; ++k) {
      if (is_complex) {
        fin >> i >> j >> re >> im;
      } else{
        fin >> i >> j >> re;
        im = 0.0;
      }
      if (fin.fail()) {
        throw std::runtime_error("Error reading matrix entry at line " + std::to_string(k + 2));
      }
      --i; --j;
      std::complex<double> val(re, im);
      entries.emplace_back(i, j, val);
      if (is_symmetric && i != j) {
        entries.emplace_back(j, i, val);
      }
      if (is_hermitian && i != j) {
        entries.emplace_back(j, i, std::conj(val));
      }
    }
  }

  CSRMatrixBlock coo_to_csr(std::size_t n_global,
                            std::size_t row_start,
                            std::size_t row_end,
                            const std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>>& coo_entries) {
    std::size_t local_nrows = row_end - row_start;
    // フィルタリング（このrankの担当する行のみ抽出）
    std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>> local_entries;
    for (const auto& [i, j, val] : coo_entries) {
      if (i >= row_start && i < row_end) {
        local_entries.emplace_back(i - row_start, j, val); // ローカル行番号に変換
      }
    }
    // 行内で列番号を昇順にソート
    std::stable_sort(local_entries.begin(), local_entries.end(),
                     [](const auto& a, const auto& b) {
                       return std::tie(std::get<0>(a), std::get<1>(a)) <
                              std::tie(std::get<0>(b), std::get<1>(b));
                     });
    // CSR形式の構築
    std::size_t local_nnz = local_entries.size();
    std::vector<std::size_t> row_ptr(local_nrows+1, 0);
    std::vector<std::size_t> col_idx(local_nnz);
    std::vector<std::complex<double>> values(local_nnz);
    // 各行に含まれる非ゼロ要素数をカウント（i行目はrow_ptr[i+1]に格納）
    for (const auto& [i, j, val] : local_entries) {
      ++row_ptr[i + 1];
    }
    // row_ptr を累積和に変換（row_ptr[i] = 行iの開始インデックス）
    for (std::size_t i=1; i <= local_nrows; ++i) {
      row_ptr[i] += row_ptr[i - 1];
    }
    // offset[i] を用意して、行ごとの現在の書き込み位置を追跡
    std::vector<std::size_t> offset = row_ptr;
    // データを正しい位置に格納
    for (const auto& [i, j, val] : local_entries) {
      std::size_t pos = offset[i]++;
      col_idx[pos] = j;
      values[pos] = val;
    }

    return CSRMatrixBlock{
      .n_global = n_global,
      .row_start = row_start, .row_end = row_end,
      .row_ptr = std::move(row_ptr), .col_idx = std::move(col_idx),
      .values = std::move(values)
    };
  }

  CSRMatrixBlock load_mm_csr(const std::string& filename) {
    std::size_t n_global = 0;
    std::size_t row_start, row_end;
    std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>> coo_entries;
    load_mm_coo(filename, n_global, coo_entries);
    std::tie(row_start, row_end) = get_row_range(n_global, 0, 1);
    return coo_to_csr(n_global, row_start, row_end, coo_entries);
  }

  CSRMatrixBlock load_distributed_mm_csr(const std::string& filename,
                                         int rank, int size, MPI_Comm comm=MPI_COMM_WORLD) {
    std::size_t n_global = 0;
    std::size_t row_start, row_end;
    std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>> local_entries;

    // Rank 0: 読み込みと分配, rank \neq 0: 受信
    if (rank == 0) {
      // 行列全体を読み込み、COO形式にする
      std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>> coo_entries;
      load_mm_coo(filename, n_global, coo_entries);
      // 全プロセスに行列のサイズを共有
      MPI_Bcast(&n_global, 1, MPI_CXX_SIZE_T, 0, comm);
      // 各プロセスへ分配を行う
      MPI_Barrier(comm);
      for (int r = 1; r < size; ++r) {
        auto [rs, re] = get_row_range(n_global, r, size);
        std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>> temp;
        for (const auto& [i, j, v] : coo_entries) {
          if (i >= rs && i < re) temp.emplace_back(i, j, v);
        }
        std::size_t count = temp.size();
        MPI_Send(&count, 1, MPI_CXX_SIZE_T, r, 0, comm);
        if (count > 0) {
          std::vector<std::size_t> is(count), js(count);
          std::vector<std::complex<double>> vs(count);
          for (std::size_t k = 0; k < count; ++k) {
            is[k] = std::get<0>(temp[k]);
            js[k] = std::get<1>(temp[k]);
            vs[k] = std::get<2>(temp[k]);
          }
          MPI_Send(is.data(), count, MPI_CXX_SIZE_T, r, 1, comm);
          MPI_Send(js.data(), count, MPI_CXX_SIZE_T, r, 2, comm);
          MPI_Send(vs.data(), count, MPI_CXX_DOUBLE_COMPLEX, r, 3, comm);
        }
      }
      // 自分の担当する行範囲を計算し、抽出
      std::tie(row_start, row_end) = get_row_range(n_global, 0, size);
      for (const auto& [i, j, v] : coo_entries) {
        if (i >= row_start && i < row_end) local_entries.emplace_back(i, j, v);
      }
    } else {
      // 行列のサイズを受信し、担当する行範囲を計算
      MPI_Bcast(&n_global, 1, MPI_CXX_SIZE_T, 0, comm);
      std::tie(row_start, row_end) = get_row_range(n_global, rank, size);
      // 担当する行列要素の情報を受信
      MPI_Barrier(comm);
      std::size_t count = 0;
      MPI_Recv(&count, 1, MPI_CXX_SIZE_T, 0, 0, comm, MPI_STATUS_IGNORE);
      if (count > 0) {
        std::vector<std::size_t> is(count), js(count);
        std::vector<std::complex<double>> vs(count);
        MPI_Recv(is.data(), count, MPI_CXX_SIZE_T,         0, 1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(js.data(), count, MPI_CXX_SIZE_T,         0, 2, comm, MPI_STATUS_IGNORE);
        MPI_Recv(vs.data(), count, MPI_CXX_DOUBLE_COMPLEX, 0, 3, comm, MPI_STATUS_IGNORE);
        local_entries.reserve(count);
        for (std::size_t k = 0; k < count; ++k) {
          local_entries.emplace_back(is[k], js[k], vs[k]);
        }
      }
    }

    return coo_to_csr(n_global, row_start, row_end, local_entries);
  }

  std::pair<std::size_t, std::size_t> get_row_range(std::size_t n_global, int rank, int size) {
    std::size_t rows_per_rank = n_global / size;
    std::size_t remainder     = n_global % size;
    std::size_t row_start, row_end;
    if (static_cast<std::size_t>(rank) < remainder) {
      row_start = rank * (rows_per_rank + 1);
      row_end   = row_start + (rows_per_rank + 1);
    } else {
      row_start = rank * rows_per_rank + remainder;
      row_end   = row_start + rows_per_rank;
    }
    return {row_start, row_end};
  }

  void SpMV(const CSRMatrixBlock& A,
            const std::vector<std::complex<double>>& x_global,
            std::vector<std::complex<double>>& y_local) {
    std::size_t local_nrows = A.row_end - A.row_start;
#pragma omp parallel for schedule(dynamic)
    for (std::size_t i = 0; i < local_nrows; ++i) {
      std::complex<double> sum = {0.0, 0.0};
      for (std::size_t j = A.row_ptr[i]; j < A.row_ptr[i+1]; ++j) {
        sum += A.values[j] * x_global[A.col_idx[j]];
      }
      y_local[i] = sum;
    }
  }

  void Timer::start() {
    running_ = true;
    start_time_ = std::chrono::high_resolution_clock::now();
  }
  void Timer::stop() {
    end_time_ = std::chrono::high_resolution_clock::now();
    running_ = false;
  }
  double Timer::elapsed_ms() const {
    auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    return std::chrono::duration<double, std::milli>(end - start_time_).count();
  }
  double Timer::elapsed_sec() const {
    auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
    return std::chrono::duration<double>(end - start_time_).count();
  }
}
