#include "common.hpp"
#include "dpu_transfer_helper.hpp"
#include "gemvf_kernel.hpp"

void print_output(dpu_set_t set) {
  dpu_set_t dpu;
  DPU_FOREACH(set, dpu) { dpu_log_read(dpu, stdout); }
}

extern "C" {
int gemv_f_basic(uint32_t m, uint32_t n, const float *mat, const float *vec, float *out) {
  Kernel kernel;

  uint32_t numDPUs = 64;  // number of available DPUs
  uint32_t rowsPerDPU;
  gemv_launch_statistics<float>(m, n, numDPUs, rowsPerDPU);

  dpu_set_t dpu_set;
  DPU_ASSERT(dpu_alloc(numDPUs, nullptr, &dpu_set));

  kernel.set_dpu_set(dpu_set, numDPUs);
  kernel.load_program("gemv_f.kernel");

  uint32_t metadata[2] = {rowsPerDPU, n};

  kernel.set_arg_broadcast("metadata", 0, metadata, sizeof(uint32_t) * 2, false);

  size_t A_offset = 0;
  size_t x_offset = alignUp(rowsPerDPU * n * sizeof(float), 8);
  size_t result_offset = x_offset + alignUp(n * sizeof(float), 8);

  kernel.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, A_offset, mat, rowsPerDPU * n * sizeof(float),
                         m * n * sizeof(float), false);

  kernel.set_arg_broadcast(DPU_MRAM_HEAP_POINTER_NAME, x_offset, vec, n * sizeof(float), false);

  kernel.launch(false);

  kernel.get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, result_offset, out, rowsPerDPU * sizeof(float), m * sizeof(float),
                        false);

  kernel.free_dpus();
  return 0;
}

int gemv_f(uint32_t m, uint32_t n, const float *A, const float *x, float *y, const float *alpha, const float *beta) {
  GEMVF_Kernel kernel;
  kernel.init(m, n);
  kernel.set_params(alpha, beta, false);
  kernel.set_A(A, false);
  kernel.set_x(x, false);
  kernel.set_y(y, false);
  kernel.launch(false);
  kernel.get_y(y, false);
  kernel.free_dpus();
  return 0;
}
}
