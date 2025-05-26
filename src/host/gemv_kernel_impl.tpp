#include "dpu_transfer_helper.hpp"

template <typename inType, typename outType>
void GEMV_Kernel<inType, outType>::set_A(const inType *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, A_offset, data, rows_per_dpu * n * sizeof(inType), m * n * sizeof(inType),
                  async);
}

template <typename inType, typename outType>
void GEMV_Kernel<inType, outType>::set_x(const inType *data, bool async) {
  set_arg_broadcast(DPU_MRAM_HEAP_POINTER_NAME, x_offset, data, n * sizeof(inType), async);
}

template <typename inType, typename outType>
void GEMV_Kernel<inType, outType>::set_y(const outType *data, bool async) {
  set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(outType), m * sizeof(outType), async);
}

template <typename inType, typename outType>
void GEMV_Kernel<inType, outType>::get_y(outType *data, bool async) {
  get_arg_gather(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(outType), m * sizeof(outType), async);
}

template <typename inType, typename outType>
void GEMV_Kernel<inType, outType>::get_y_safe(outType *data) {
  get_arg_gather_safe(DPU_MRAM_HEAP_POINTER_NAME, y_offset, data, rows_per_dpu * sizeof(outType), m * sizeof(outType));
}

template <typename inType, typename outType>
void GEMV_Kernel<inType, outType>::set_params(const outType *alpha, const outType *beta, bool async) {
  params args{.rows_per_dpu = this->rows_per_dpu, .row_size = n, .alpha = *alpha, .beta = *beta};
  this->set_arg_broadcast_exact("args", 0, reinterpret_cast<uint8_t *>(&args), sizeof(params), async);
}

template <typename inType, typename outType>
bool GEMV_Kernel<inType, outType>::init(uint32_t m, uint32_t n) {
  this->nr_dpus = 64;
  gemv_launch_statistics<outType>(m, n, this->nr_dpus, this->rows_per_dpu);
  return this->init(m, n, nr_dpus, rows_per_dpu);
}

template <typename inType, typename outType>
bool GEMV_Kernel<inType, outType>::init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu) {
  this->m = m;
  this->n = n;
  this->nr_dpus = nr_dpus;
  this->rows_per_dpu = rows_per_dpu;

  if (dpu_alloc(this->nr_dpus, nullptr, &this->dpu_set) != DPU_OK) {
    return false;
  }

  this->load_program(this->program_name.c_str());

  A_offset = 0;
  x_offset = alignUp(rows_per_dpu * n * sizeof(inType), 8);
  y_offset = x_offset + alignUp(n * sizeof(inType), 8);

  return true;
}
