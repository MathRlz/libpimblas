#include "dpu_transfer_helper.hpp"

#include "common.hpp"

template <typename T>
void transfer_chunks_to_mram(dpu_set_t set, const char *symbol, T *data, size_t chunk_size, size_t size) {
  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  transfer_chunks<T>(set, nr_dpus, DPU_XFER_TO_DPU, DPU_XFER_DEFAULT, 
                            symbol, 0, data, chunk_size, size);
}

template <typename T>
void transfer_full_to_mram(dpu_set_t set, const char *symbol, T *data, size_t size) {
  DPU_ASSERT(dpu_broadcast_to(set, symbol, 0, data, alignUp(size * sizeof(T), 8), DPU_XFER_DEFAULT));
}

template <typename T>
void transfer_chunks_from_mram(dpu_set_t set, const char *symbol, T *data, size_t chunk_size, size_t size) {
  uint32_t nr_dpus = 0;
  DPU_ASSERT(dpu_get_nr_dpus(set, &nr_dpus));
  transfer_chunks<T>(set, nr_dpus, DPU_XFER_FROM_DPU, DPU_XFER_DEFAULT, 
                            symbol, 0, data, chunk_size, size);
}

template <typename T>
size_t transfer_chunks_from_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t chunkSize,
                                          size_t size) {
  return transfer_chunks<T>(set, nrDPUs, DPU_XFER_FROM_DPU, DPU_XFER_DEFAULT, 
                            DPU_MRAM_HEAP_POINTER_NAME, offset, data, chunkSize, size);
}

template <typename T>
size_t transfer_chunks_to_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t chunkSize,
                                        size_t size) {
  return transfer_chunks<T>(set, nrDPUs, DPU_XFER_TO_DPU, DPU_XFER_DEFAULT, 
                            DPU_MRAM_HEAP_POINTER_NAME, offset, data, chunkSize, size);
}

template <typename T>
size_t transfer_full_to_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t size) {
  size_t copySize = alignUp(size * sizeof(T), 8);
  DPU_ASSERT(dpu_broadcast_to(set, DPU_MRAM_HEAP_POINTER_NAME, offset, data, copySize, DPU_XFER_DEFAULT));
  return offset + copySize;
}

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

template size_t transfer_chunks_from_mram_directly<int>(dpu_set_t set, uint32_t nrDPUs, size_t offset, int *data,
                                                          size_t chunkSize, size_t size);
template size_t transfer_chunks_from_mram_directly<float>(dpu_set_t set, uint32_t nrDPUs, size_t offset, float *data,
                                                          size_t chunkSize, size_t size);

template size_t transfer_chunks_to_mram_directly<int>(dpu_set_t set, uint32_t nrDPUs, size_t offset, int *data,
                                                        size_t chunkSize, size_t size);
template size_t transfer_chunks_to_mram_directly<const int>(dpu_set_t set, uint32_t nrDPUs, size_t offset, const int *data,
                                                        size_t chunkSize, size_t size);

template size_t transfer_chunks_to_mram_directly<float>(dpu_set_t set, uint32_t nrDPUs, size_t offset, float *data,
                                                        size_t chunkSize, size_t size);
template size_t transfer_chunks_to_mram_directly<const float>(dpu_set_t set, uint32_t nrDPUs, size_t offset,
                                                              const float *data, size_t chunkSize, size_t size);

template size_t transfer_full_to_mram_directly<int>(dpu_set_t set, uint32_t nrDPUs, size_t offset, int *data,
                                                      size_t size);
template size_t transfer_full_to_mram_directly<const int>(dpu_set_t set, uint32_t nrDPUs, size_t offset,
                                                            const int *data, size_t size);

template size_t transfer_full_to_mram_directly<float>(dpu_set_t set, uint32_t nrDPUs, size_t offset, float *data,
                                                      size_t size);
template size_t transfer_full_to_mram_directly<const float>(dpu_set_t set, uint32_t nrDPUs, size_t offset,
                                                            const float *data, size_t size);

template void transfer_full_to_mram<uint32_t>(dpu_set_t set, const char *symbol, uint32_t *data, size_t size);
template void transfer_full_to_mram<uint8_t>(dpu_set_t set, const char *symbol, uint8_t *data, size_t size);