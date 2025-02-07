#include "dpu_transfer_helper.hpp"
#include "gemvInt32_kernel.hpp"

extern "C" {
int gemv_int32(uint32_t m, uint32_t n, const int *A, const int *x, int *y, const int *alpha, const int *beta) {
  GEMVInt32_Kernel_Beta kernel;
  kernel.init(m, n);
  kernel.set_params(alpha, beta, false);
  kernel.set_A(A, true);
  kernel.set_x(x, true);
  kernel.set_y(y, true);
  kernel.launch(true);
  kernel.get_y(y, true);
  kernel.sync();
  kernel.free_dpus();
  return 0;
}
}