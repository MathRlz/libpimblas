#include "common.hpp"
#include "test_helper.hpp"

int host_gemv_f(uint32_t m, uint32_t n, const float *mat, const float *vec, float *y, float alpha, float beta) {
  for (size_t row = 0; row < m; ++row) {
    float mul_res = 0.0f;
    for (size_t col = 0; col < n; ++col) {
      mul_res += vec[col] * mat[row * n + col];
    }
    y[row] = alpha * mul_res + y[row] * beta;
  }

  return 0;
}

int host_gemv_f_basic(uint32_t m, uint32_t n, const float *mat, const float *vec, float *y) {
  for (size_t row = 0; row < m; ++row) {
    float mul_res = 0.0f;
    for (size_t col = 0; col < n; ++col) {
      mul_res += vec[col] * mat[row * n + col];
    }
    y[row] = mul_res;
  }

  return 0;
}

int main(int argc, char **argv) {
  const int M = 4;
  const int N = 4;
  auto mat = generateRandomFloats(M * N, 1.0f, 1.0f);
  auto vec = generateRandomFloats(N, 1.0f, 1.0f);
  auto y = generateRandomFloats(M, 1.0f, 1.0f);
  auto y_host = pimblas::vector<float>(y.begin(), y.end());
  float alpha = 0.5f;
  float beta = 2.0f;

  int ret = 0;
  if ((ret = gemv_f(M, N, mat.data(), vec.data(), y.data(), &alpha, &beta)) != 0) {
    RET_TEST_FAIL;
  }

  if ((ret = host_gemv_f(M, N, mat.data(), vec.data(), y_host.data(), alpha, beta)) != 0) {
    RET_TEST_FAIL;
  }

  // 0.01 percent difference at most
  bool same = mostly_same_rel(y.data(), y_host.data(), M, 1e-4f);
  if (same) {
    std::cout << "SUCCESS " << std::endl;
    RET_TEST_OK;
  }

  RET_TEST_FAIL;
}