#include "matrix_transpose.hpp"

#include <immintrin.h>

#include <algorithm>

void transpose8x8_avx(const float *src, float *dst, size_t src_cols, size_t dst_rows) {
  auto col0 = _mm256_loadu_ps(&src[0 * src_cols]);
  auto col1 = _mm256_loadu_ps(&src[1 * src_cols]);
  auto col2 = _mm256_loadu_ps(&src[2 * src_cols]);
  auto col3 = _mm256_loadu_ps(&src[3 * src_cols]);
  auto col4 = _mm256_loadu_ps(&src[4 * src_cols]);
  auto col5 = _mm256_loadu_ps(&src[5 * src_cols]);
  auto col6 = _mm256_loadu_ps(&src[6 * src_cols]);
  auto col7 = _mm256_loadu_ps(&src[7 * src_cols]);

  auto t0 = _mm256_unpacklo_ps(col0, col1);  // a00, a10, a01, a11 ...
  auto t1 = _mm256_unpackhi_ps(col0, col1);  // a02, a12, a03, a13
  auto t2 = _mm256_unpacklo_ps(col2, col3);
  auto t3 = _mm256_unpackhi_ps(col2, col3);
  auto t4 = _mm256_unpacklo_ps(col4, col5);
  auto t5 = _mm256_unpackhi_ps(col4, col5);
  auto t6 = _mm256_unpacklo_ps(col6, col7);
  auto t7 = _mm256_unpackhi_ps(col6, col7);

  auto r0 = _mm256_shuffle_ps(t0, t2, 0x44);  // a00, a10, a20, a30
  auto r1 = _mm256_shuffle_ps(t0, t2, 0xEE);  // a01, a11, a21, a31
  auto r2 = _mm256_shuffle_ps(t1, t3, 0x44);
  auto r3 = _mm256_shuffle_ps(t1, t3, 0xEE);
  auto r4 = _mm256_shuffle_ps(t4, t6, 0x44);
  auto r5 = _mm256_shuffle_ps(t4, t6, 0xEE);
  auto r6 = _mm256_shuffle_ps(t5, t7, 0x44);
  auto r7 = _mm256_shuffle_ps(t5, t7, 0xEE);

  auto p0 = _mm256_permute2f128_ps(r0, r4, 0x20);  // Low 128 bits of r0 and r4
  auto p1 = _mm256_permute2f128_ps(r1, r5, 0x20);
  auto p2 = _mm256_permute2f128_ps(r2, r6, 0x20);
  auto p3 = _mm256_permute2f128_ps(r3, r7, 0x20);
  auto p4 = _mm256_permute2f128_ps(r0, r4, 0x31);  // High 128 bits of r0 and r4
  auto p5 = _mm256_permute2f128_ps(r1, r5, 0x31);
  auto p6 = _mm256_permute2f128_ps(r2, r6, 0x31);
  auto p7 = _mm256_permute2f128_ps(r3, r7, 0x31);

  _mm256_storeu_ps(&dst[0 * dst_rows], p0);
  _mm256_storeu_ps(&dst[1 * dst_rows], p1);
  _mm256_storeu_ps(&dst[2 * dst_rows], p2);
  _mm256_storeu_ps(&dst[3 * dst_rows], p3);
  _mm256_storeu_ps(&dst[4 * dst_rows], p4);
  _mm256_storeu_ps(&dst[5 * dst_rows], p5);
  _mm256_storeu_ps(&dst[6 * dst_rows], p6);
  _mm256_storeu_ps(&dst[7 * dst_rows], p7);
}

void transpose_matrix_column_major(const float *src, float *dst, size_t rows, size_t cols) {
  constexpr size_t block_size = 8;

  for (size_t i = 0; i < cols; i += block_size) {
    for (size_t j = 0; j < rows; j += block_size) {
      size_t block_rows = std::min(block_size, rows - j);
      size_t block_cols = std::min(block_size, cols - i);

      if (block_rows == 8 && block_cols == 8) {
        transpose8x8_avx(&src[i * rows + j], &dst[j * cols + i], rows, cols);
      } else {
        for (size_t ii = 0; ii < block_cols; ii++) {
          for (size_t jj = 0; jj < block_rows; jj++) {
            dst[(j + jj) * cols + (i + ii)] = src[(i + ii) * rows + (j + jj)];
          }
        }
      }
    }
  }
}