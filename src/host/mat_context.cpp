#include "mat_context.hpp"

#include "common.hpp"

extern "C" {
int create_context(dpu_mat_handle_t *dpu_context) {
  *dpu_context = reinterpret_cast<dpu_mat_handle_t>(new dpu_mat());
  return 0;
}

int initialize_context(dpu_mat_handle_t dpu_context, const float *matrix, unsigned int rows, unsigned int cols) {
  return reinterpret_cast<dpu_mat *>(dpu_context)->initialize(matrix, rows, cols);
}

int multiply_by_vector(dpu_mat_handle_t dpu_context, const float *vecIn, float *vecOut) {
  return reinterpret_cast<dpu_mat *>(dpu_context)->gemv(vecIn, vecOut);
}

int destroy_context(dpu_mat_handle_t dpu_context) {
  delete reinterpret_cast<dpu_mat *>(dpu_context);
  return 0;
}
}

dpu_mat::dpu_mat() {}

dpu_mat::~dpu_mat() {
  if (kernel) {
    kernel->free_dpus();
  }
}

int dpu_mat::initialize(const float *matrix, uint32_t rows, uint32_t cols) {
  kernel = std::unique_ptr<GEMVF_Kernel>(new GEMVF_Kernel());
  kernel->init(rows, cols);
  float alpha = 1.0f;
  kernel->set_params(&alpha, true);
  kernel->set_A(matrix, true);
  kernel->sync();
  return 0;
}

int dpu_mat::gemv(const float *vecIn, float *vecOut) {
  kernel->set_x(vecIn, true);
  kernel->launch(true);
  kernel->get_y(vecOut, true);
  kernel->sync();
  return 0;
}
