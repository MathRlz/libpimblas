#include <chrono>

#include "common.hpp"
#include "test_helper.hpp"

int host_sgemm_int32_column_major(const int *A, const int *B, int *C, int alpha, int beta, uint32_t M, uint32_t N,
                                  uint32_t K) {
  // Loop over columns of C
  for (size_t col = 0; col < N; col++) {
    // Loop over rows of C
    for (size_t row = 0; row < M; row++) {
      float sum = 0.0f;
      for (size_t i = 0; i < K; i++) {
        sum += A[row + i * M] * B[i + col * K];
      }
      C[row + col * M] = alpha * sum + beta * C[row + col * M];
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  const int M = 8;
  const int N = 8;
  const int K = 8;
  auto A = generateRandomIntegers(M * K, 1, 10);
  auto B = generateRandomIntegers(K * N, 1, 10);
  auto C = generateRandomIntegers(M * N, 1, 10);
  auto C_host = pimblas::vector<int>(C.begin(), C.end());
  int alpha = 1;
  int beta = 1;

  sgemm_int32(M, K, N, A.data(), B.data(), C.data(), &alpha, &beta);

  host_sgemm_int32_column_major(A.data(), B.data(), C_host.data(), alpha, beta, M, N, K);

  bool same = same_vectors(C, C_host);
  if (same) {
    std::cout << "SUCCESS " << std::endl;
    RET_TEST_OK;
  }

  RET_TEST_FAIL;
}