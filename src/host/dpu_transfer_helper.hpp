#include <cstddef>
#include <cstdint>

struct dpu_set_t;

template <typename T>
T alignUp(T value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

template <typename T>
void transfer_chunks_to_mram(dpu_set_t set, const char *symbol, T *data, size_t chunk_size, size_t size);

template <typename T>
void transfer_full_to_mram(dpu_set_t set, const char *symbol, T *data, size_t size);

template <typename T>
void transfer_chunks_from_mram(dpu_set_t set, const char *symbol, T *data, size_t chunk_size, size_t size);

template <typename T>
size_t transfer_chunks_from_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t chunkSize,
                                          size_t size);

template <typename T>
size_t transfer_chunks_to_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t chunkSize,
                                        size_t size);

template <typename T>
size_t transfer_full_to_mram_directly(dpu_set_t set, uint32_t nrDPUs, size_t offset, T *data, size_t size);

template <typename T>
void gemv_launch_statistics(uint32_t m, uint32_t n, uint32_t &numDPUs, uint32_t &rowsPerDPU);