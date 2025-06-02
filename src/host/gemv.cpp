#include "common.hpp"
#include "gemv_kernel.hpp"

template <typename inType, typename outType, class Kernel>
int gemv(uint32_t m, uint32_t n, const inType *mat, const inType *vec, outType *out, const outType *alpha,
         const outType *beta) {
  Kernel kernel;
  if (kernel.init(m, n) == false) {
    show_error("gemv: Couldn't initialize kernel for m=[{}] n=[{}]", m, n);
    return -1;
  }
  kernel.set_params(alpha, beta, false);
  kernel.set_A(mat, true);
  kernel.set_x(vec, true);
  if (*beta != 0) {
    kernel.set_y(out, true);
  }
  kernel.launch(true);
  kernel.get_y(out, true);
  kernel.sync();
  return 0;
}

extern "C" {
int gemv_int8(uint32_t m, uint32_t n, const int8_t *A, const int8_t *x, int *y, const int *alpha, const int *beta) {
  return gemv<int8_t, int, GEMV_INT8_Kernel>(m, n, A, x, y, alpha, beta);
}

int gemv_int32(uint32_t m, uint32_t n, const int *A, const int *x, int *y, const int *alpha, const int *beta) {
  return gemv<int, int, GEMV_INT32_Kernel>(m, n, A, x, y, alpha, beta);
}

int gemv_f_basic(uint32_t m, uint32_t n, const float *mat, const float *vec, float *out) {
  float alpha = 1.0f;
  float beta = 0.0f;
  return gemv<float, float, GEMVF_Kernel>(m, n, mat, vec, out, &alpha, &beta);
}

int gemv_f(uint32_t m, uint32_t n, const float *A, const float *x, float *y, const float *alpha, const float *beta) {
  return gemv<float, float, GEMVF_Kernel>(m, n, A, x, y, alpha, beta);
}
}
