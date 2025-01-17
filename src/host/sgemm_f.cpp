#include <cassert>

#include "common.hpp"
#include "dpu_transfer_helper.hpp"

template <typename T>
void transposeMatrix(const T *matrix, T *result, size_t rows, size_t cols) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      result[j * rows + i] = matrix[i * cols + j];
    }
  }
}

template <typename T>
void convertRowToColumnMajor(const T *rowMajor, T *columnMajor, size_t rows, size_t cols, size_t ld) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      columnMajor[j * rows + i] = rowMajor[i * ld + j];
    }
  }
}

template <typename T>
void convertColumnToRowMajor(const T *columnMajor, T *rowMajor, size_t rows, size_t cols, size_t ld) {
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      rowMajor[i * cols + j] = columnMajor[j * ld + i];
    }
  }
}

void calculate_single_row(dpu_set_t set, uint32_t numDPUs, uint32_t rowsPerDPU, uint32_t m, uint32_t n, size_t x_offset,
                          const float *vecB, float *vecC) {
  size_t y_offset = transfer_full_to_mram_directly(set, numDPUs, x_offset, vecB, n);
  transfer_chunks_to_mram_directly(set, numDPUs, y_offset, vecC, rowsPerDPU, m);

  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  transfer_chunks_from_mram_directly(set, numDPUs, y_offset, vecC, rowsPerDPU, m);
}

// Assumption A is in row order, B and C are in column order
// A is of size rowsA x rowsB
// B rowsB x colsB
// C rowsA x rowsB
void sgemm_f(uint32_t rowsA, uint32_t rowsB, uint32_t colsB, const float *A, const float *B, float *C, float alpha,
             float beta) {
  uint32_t numDPUs = 64;
  uint32_t rowsPerDPU;
  gemv_launch_statistics<float>(rowsA, rowsB, numDPUs, rowsPerDPU);

  dpu_set_t set;
  DPU_ASSERT(dpu_alloc(numDPUs, nullptr, &set));

  char *kernName = pimblas_get_kernel_dir_concat_free("gemv_f_y.kernel");
  show_debug("kern_path = {}", kernName);
  DPU_ASSERT(dpu_load(set, kernName, nullptr));
  free(kernName);

  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    float alpha;
    float beta;
  };

  params args = {.rows_per_dpu = rowsPerDPU, .row_size = rowsB, .alpha = alpha, .beta = beta};

  transfer_full_to_mram(set, "args", reinterpret_cast<uint8_t *>(&args), sizeof(args));

  size_t A_offset = 0;
  size_t x_offset = transfer_chunks_to_mram_directly(set, numDPUs, 0, A, rowsPerDPU * rowsB, rowsA * rowsB);

  for (uint32_t i = 0; i < colsB; i++) {
    size_t y_offset = transfer_full_to_mram_directly(set, numDPUs, x_offset, B + rowsB * i, rowsB);
    transfer_chunks_to_mram_directly(set, numDPUs, y_offset, C + rowsA * i, rowsPerDPU, rowsA);

    DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

    transfer_chunks_from_mram_directly(set, numDPUs, y_offset, C + rowsA * i, rowsPerDPU, rowsA);
  }

  DPU_ASSERT(dpu_free(set));
}

bool is_transpose(char trans) {
  if (trans == 'N' || trans == 'n') {
    return false;
  } else if (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c') {
    return true;
  }
  // Unknown trans
  return false;
}

// Performs: C = alpha * op(A) * op(B) + beta * C
// a, b, c are always in column major order
// trans - if 'n' op(a) = A, if 't' or 'c' op (A) = A**T
// ld - leading dimension as declared in the calling program.
// op(A) is an m by k matrix
// op(B) is a k by n matrix
// C is an m by n matrix

void sgemm_wrapper(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
                   const float *a, const int *lda, const float *b, const int *ldb, const float *beta, float *c,
                   const int *ldc) {
  const float *a_buffer = nullptr;
  float *a_tmp_buffer = nullptr;

  if (false == is_transpose(*transa)) {
    // Matrix is in colum major order
    // And is treated as is that means
    // we need to change it into row major in order to use
    // it with our algorithm.
    assert(*lda == *m && "Unexpected padding in matrix A - UNHANDLED");
    a_tmp_buffer = reinterpret_cast<float *>(malloc(alignUp(*m * *k * sizeof(float), 8)));

    convertColumnToRowMajor(a, a_tmp_buffer, *m, *k, *lda);

    a_buffer = a_tmp_buffer;
  } else {
    // Matrix is in column major order
    // And is treated as transposed
    // that means we can treat it as is already
    // assuming there's no padding
    // Just switch the dimensions
    assert(*lda == *k && "Unexpected padding in matrix A - UNHANDLED");
    a_buffer = a;
  }

  const float *b_buffer = nullptr;
  float *b_tmp_buffer = nullptr;
  if (false == is_transpose(*transb)) {
    // No need to do anything
    assert(*ldb == *k && "Unexpected padding in matrix B - UNHANDLED");
    b_buffer = b;
  } else {
    // Matrix is in col major ordering and is treated as transposed
    // That means if we convert it then it's all good
    // we got B (LDB/n, k) and we need it to be B**T (k, n)
    assert(*ldb == *n && "Unexpected padding in matrix B - UNHANDLED");
    b_tmp_buffer = reinterpret_cast<float *>(malloc(alignUp(*n * *k * sizeof(float), 8)));
    convertColumnToRowMajor(b, b_tmp_buffer, *n, *k, *k);
    b_buffer = b_tmp_buffer;
  }

  // C is already in column major order no need to do anything
  assert(*ldc == *m && "Unexpected padding in matrix C - UNHANDLED");

  sgemm_f(*m, *k, *n, a_buffer, b_buffer, c, *alpha, *beta);

  free(a_tmp_buffer);
  free(b_tmp_buffer);
}
