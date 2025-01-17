#include "common.hpp"
#include "dpu_transfer_helper.hpp"

extern "C" {
int gemv_int32(uint32_t m, uint32_t n, const int *A, const int *x, int *y, const int *alpha, const int *beta) {
    struct params {
        uint32_t rows_per_dpu;
        uint32_t row_size;
        int alpha;
        int beta;
  };

  uint32_t numDPUs = 64;
  uint32_t rowsPerDPU;
  gemv_launch_statistics<int>(m, n, numDPUs, rowsPerDPU);

  show_trace("gemv_f m={}, n={}, A={}, x={}, y={}, alpha={}, beta={}, numDPUs={}, rowsPerDPU={}", m, n,
             reinterpret_cast<const uintptr_t>(A), reinterpret_cast<const uintptr_t>(x),
             reinterpret_cast<const uintptr_t>(y), *alpha, *beta, numDPUs, rowsPerDPU);

  dpu_set_t set;
  DPU_ASSERT(dpu_alloc(numDPUs, nullptr, &set));

  char *kernName = pimblas_get_kernel_dir_concat_free("gemv_int32.kernel");
  show_debug("kern_path = {}", kernName);
  DPU_ASSERT(dpu_load(set, kernName, nullptr));
  free(kernName);

  params args = {.rows_per_dpu = rowsPerDPU, .row_size = n, .alpha = *alpha, .beta = *beta};

  transfer_full_to_mram(set, "args", reinterpret_cast<uint8_t *>(&args), sizeof(args));

  size_t offset = 0;
  offset = transfer_chunks_to_mram_directly(set, numDPUs, offset, A, rowsPerDPU * n, m * n);
  offset = transfer_full_to_mram_directly(set, numDPUs, offset, x, n);
  transfer_chunks_to_mram_directly(set, numDPUs, offset, y, rowsPerDPU, m);

  DPU_ASSERT(dpu_launch(set, DPU_SYNCHRONOUS));

  // print_output(set);

  transfer_chunks_from_mram_directly(set, numDPUs, offset, y, rowsPerDPU, m);

  DPU_ASSERT(dpu_free(set));

  return 0;
}
}