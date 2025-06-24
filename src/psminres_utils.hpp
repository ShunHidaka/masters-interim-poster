#ifndef PSMINRES_UTIL_HPP
#define PSMINRES_UTIL_HPP

#include <chrono>
#include <complex>
#include <iostream>
#include <utility>
#include <string>
#include <tuple>
#include <vector>
#include "mpi_raii.hpp"
//#include <mpi.h>

namespace utils {

  /**
   */
  void set_shift(std::size_t& M, std::vector<std::complex<double>>& shift);

  /**
   * \brief 行分割可能なCompressed Sparse Row 形式
   */
  struct CSRMatrixBlock {
    // 行列全体
    std::size_t n_global; ///< 行列のサイズ（全体の行数=全体の列数）
    // [row_start, row_end)
    std::size_t row_start;    ///< このプロセスが担当する行の開始添え字
    std::size_t row_end;      ///< このプロセスが担当する行の終端添え字
    // CSR形式本体
    std::vector<std::size_t> row_ptr;         ///< 行ポインタ（size=local_nrows+1）
    std::vector<std::size_t> col_idx;         ///< 列インデックス（size=非ゼロ要素）
    std::vector<std::complex<double>> values; ///< 対応する値（size=非ゼロ要素）
  };
  /**
   * \brief Matrix Market形式を読み込み、COO形式に変換する
   */
  void load_mm_coo(const std::string& filename,
                   std::size_t& matrix_size,
                   std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>>& entries);
  /**
   * \brief COO形式をCSR形式に変換する
   */
  CSRMatrixBlock coo_to_csr(std::size_t n_global,
                            std::size_t row_start,
                            std::size_t row_end,
                            const std::vector<std::tuple<std::size_t, std::size_t, std::complex<double>>>& coo_entries);

  /**
   * \brief Matrix Market形式の読み込み、CSR形式で返す（内部でload_mm_coo, coo_to_csr を使用）
   */
  CSRMatrixBlock load_mm_csr(const std::string& filename);

  /**
   * \brief load_mm_csr で行分割をおこなう
   */
  CSRMatrixBlock load_distributed_mm_csr(const std::string& filename,
                                         int rank, int size, MPI_Comm comm);
  /**
   * \brief 与えられたrank に応じた行分割の割り当て範囲を返す
   */
  std::pair<std::size_t, std::size_t> get_row_range(std::size_t n_global, int rank, int size);

  /**
   * \brief 疎行列ベクトル積関数（引数で全体のベクトルを受け取る）
   */
  void SpMV(const CSRMatrixBlock& A,
            const std::vector<std::complex<double>>& x_global,
            std::vector<std::complex<double>>& y_local);

  // 各プロセスが個別にファイルを読み込んでCSR形式の行列を用意する
  // ファイル競合であったり自分の行まで読み飛ばすだったりが必要
  // distributed よりも負荷が均等にはできそう
  void load_parallel_mm_csr();

  /**
   * \brief 実行時間測定用クラス
   *
   * 挙動について：
   * 1. MPI_Init 前に Timer を宣言・start()
   *    MPI_Finalize 後に Timer の stop()
   *    ではMPIを含む全体の実行時間を測定
   * 2. MPI_Init 前に Timer を宣言・start()
   *    MPI_Finalize 前に Timer の stop()
   *    では各プロセスで個別に Timer が作成され
   *    MPI_Finalize を受けて個別に削除される
   *
   * MPI対応：
   * std::chrono::high_resolution_clock::now() => MPI_Wtime
   * std::chrono::high_resolution_clock::time_point => double
   * に入れ替えればMPIを使用したクラスにできる
   * この場合はMPI内でしか使用不可になる
   */
  class Timer {
  public:
    void start();
    void stop();
    double elapsed_ms() const; ///< ミリ秒単位
    double elapsed_sec() const; ///< 秒単位
  private:
    std::chrono::high_resolution_clock::time_point start_time_;
    std::chrono::high_resolution_clock::time_point end_time_;
    bool running_ = false;
  };
}


#endif // PSMINRES_UTIL_HPP

// Parallelized Matrix-Vector mulitiplication REFERENCES
// https://www.cs.hunter.cuny.edu/~sweiss/course_materials/csci493.65/lecture_notes_2014/chapter08.pdf
// https://edoras.sdsu.edu/~mthomas/sp17.605/lectures/MPI-MatrixVectorMult.pdf
// https://arxiv.org/abs/1812.00904
// https://arxiv.org/abs/1006.2183
