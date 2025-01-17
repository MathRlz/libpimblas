#include "common.hpp"
#include "test_helper.hpp"

int host_gemv_int32(uint32_t m, uint32_t n, const int *mat, const int *vec, int *y, int alpha, int beta) {
  for (size_t row = 0; row < m; ++row) {
    float mul_res = 0.0f;
    for (size_t col = 0; col < n; ++col) {
      mul_res += vec[col] * mat[row * n + col];
    }
    y[row] = alpha * mul_res + y[row] * beta;
  }

  return 0;
}

int main(int argc, char **argv) {
  const int M = 1024;
  const int N = 15000;
  auto mat = generateRandomIntegers(M * N, 1, 10);
  auto vec = generateRandomIntegers(N, 1, 10);
  auto y = generateRandomIntegers(M, 1, 10);
  auto y_host = pimblas::vector<int>(y.begin(), y.end());
  int alpha = 1;
  int beta = 2;

  int ret = 0;
  if ((ret = gemv_int32(M, N, mat.data(), vec.data(), y.data(), &alpha, &beta)) != 0) {
    RET_TEST_FAIL;
  }

  if ((ret = host_gemv_int32(M, N, mat.data(), vec.data(), y_host.data(), alpha, beta)) != 0) {
    RET_TEST_FAIL;
  }

  bool same = same_vectors(y, y_host);
  if (same) {
    std::cout << "SUCCESS " << std::endl;
    RET_TEST_OK;
  }

  RET_TEST_FAIL;
}