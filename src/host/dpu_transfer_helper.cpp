#include "dpu_transfer_helper.hpp"

#include "common.hpp"

template <typename T>
void gemv_launch_statistics(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU) {
  // Assumptions:
  // MRAM size of each DPU is 64MB
  // part of A needs to be copied to each DPU - n * rows_per_dpu
  // x vector needs to be copied to each DPU - n
  // part of y vector needs to be copied to each DPU - rows_per_dpu
  // Total ints per DPU: n * (rows_per_dpu + 1) + rows_per_dpu
  // Threads per DPU: 16
  // At minimum two rows per tasklet when sizeof(T) == 4 (because the output needs to be 8B aligned)
  constexpr size_t minRowsPerDPU = 16 * 8 / sizeof(T);

  rowsPerDPU = alignUp((m - 1) / numDPUs + 1, minRowsPerDPU);
  size_t memory_requirement = (n * (rowsPerDPU + 1) + rowsPerDPU) * sizeof(T);

  // Let's leave 1 MB
  constexpr size_t mem_cap = 63 * 1024 * 1024;
  while (memory_requirement > mem_cap) {
    rowsPerDPU -= minRowsPerDPU;
    memory_requirement = n * (rowsPerDPU + 1) + rowsPerDPU;
  }

  if (rowsPerDPU < minRowsPerDPU) {
    rowsPerDPU = minRowsPerDPU;
  }

  numDPUs = (m - 1) / rowsPerDPU + 1;
}

// Instantiation
template void gemv_launch_statistics<int>(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU);
template void gemv_launch_statistics<float>(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU);