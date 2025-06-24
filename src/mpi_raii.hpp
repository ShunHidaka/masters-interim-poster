#ifndef MPI_RAII_HPP
#define MPI_RAII_HPP

#include <iostream>
#include <stdexcept>
#include <string>
#include <mpi.h>

/// inline グローバル変数でも OK
inline MPI_Datatype MPI_CXX_SIZE_T = MPI_DATATYPE_NULL;

/**
 * \brief RAII による MPI の管理を行うクラス
 * \details
 * このクラスは、MPI の初期化と終了処理を自動的に行います。
 * コンストラクタで MPI_Init_thread を呼び出し、デストラクタで MPI_Finalize を呼び出します。
 * これにより、MPI のリソース管理を安全かつ確実に行うことができます。
 */
class MPIEnv {
public:
  /**
   * \brief コンストラクタで MPI_Init_thread を実行
   * \param argc main関数のargc
   * \param argv main関数のargc
   * \param required 要求するスレッドサポートレベル（デフォルトはMPI_THREAD_SINGLE）
   * \throws std::runtime_error MPIの初期化に失敗した場合
   */
  MPIEnv(int& argc, char**& argv, int required=MPI_THREAD_SINGLE) {
    int already_initialized = 0;
    MPI_Initialized(&already_initialized);
    if (already_initialized) {
      throw std::runtime_error("MPI is already initialized. "
                               "Multiple initialization are not allowed.");
    }

    int provided;
    int err = MPI_Init_thread(&argc, &argv, required, &provided);
    if (err != MPI_SUCCESS) {
      char err_string[MPI_MAX_ERROR_STRING];
      int len = 0;
      MPI_Error_string(err, err_string, &len);
      throw std::runtime_error(std::string("MPI_Init_thread failed: ") + err_string);
    }
    if (provided < required) {
      throw std::runtime_error("MPI thread level not sufficient");
    }

    MPI_CXX_SIZE_T = detect_mpi_size_t_type();
    initialized_ = true;
  }

  /**
   * \brief デストラクタで MPI_Finalize を呼び出す
   */
  ~MPIEnv() {
    if (initialized_) {
      int finalized = 0;
      MPI_Finalized(&finalized);
      if (!finalized) {
        MPI_Finalize();
      }
    }
  }

  // コピー・ムーブ操作を禁止
  MPIEnv(const MPIEnv&) = delete;
  MPIEnv& operator=(const MPIEnv&) = delete;

private:
  bool initialized_ = false; ///< MPI が初期化されたかどうかのフラグ

  MPI_Datatype detect_mpi_size_t_type() const {
    if (sizeof(std::size_t)      == sizeof(unsigned int))
      return MPI_UNSIGNED;
    else if (sizeof(std::size_t) == sizeof(unsigned long))
      return MPI_UNSIGNED_LONG;
    else if (sizeof(std::size_t) == sizeof(unsigned long long))
      return MPI_UNSIGNED_LONG_LONG;
    else
      throw std::runtime_error("Unsupported size_t size for MPI");
  }
};

#endif // MPI_RAII_HPP
