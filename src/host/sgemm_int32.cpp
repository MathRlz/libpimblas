#include <cassert>

#include "dpu_transfer_helper.hpp"
#include "gemvInt32_kernel.hpp"
#include "matrix_transpose.hpp"

template <typename Kernel>
Kernel &get_free_kernel(std::vector<Kernel> &kernels, size_t &cur_kernel) {
  if (cur_kernel >= kernels.size()) {
    cur_kernel = 0;
  }
  auto &kernel = kernels[cur_kernel];
  cur_kernel++;
  return kernel;
}

extern "C" {
// Assume all matricies are passed in col major order
void sgemm_int32(uint32_t rowsA, uint32_t rowsB, uint32_t colsB, const int *A_tmp, const int *B, int *C,
                 const int *alpha, const int *beta) {
  auto A = reinterpret_cast<int *>(malloc(alignUp(rowsA * rowsB * sizeof(int), 8)));
  transpose_matrix_column_major((const float *)A_tmp, (float *)A, rowsA, rowsB);

  uint32_t nr_dpus = 512;
  uint32_t rows_per_dpu = 0;
  gemv_launch_statistics<int>(rowsA, rowsB, nr_dpus, rows_per_dpu);

  auto nr_kernels = colsB;

  if (*beta == 0) {
    std::vector<GEMVInt32_Kernel> kernels(nr_kernels);
    size_t kernel_it = 0;
    for (kernel_it = 0; kernel_it < kernels.size(); kernel_it++) {
      auto &kernel = kernels[kernel_it];
      if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
        break;
      }

      kernel.set_params(alpha, false);
      kernel.set_A(A, true);
    }
    kernels.resize(kernel_it);

    show_trace("Running {} kernels. Each kernel with {} DPUs.\n", kernels.size(), nr_dpus);

    size_t cur_kernel = 0;
    for (uint32_t i = 0; i < colsB; i++) {
      auto &kernel = get_free_kernel(kernels, cur_kernel);
      kernel.set_x(B + rowsB * i, true);
      kernel.launch(true);
      kernel.get_y(C + rowsA * i, true);
    }

    for (auto &kernel : kernels) {
      kernel.sync();
      kernel.free_dpus();
    }
  } else {
    std::vector<GEMVInt32_Kernel_Beta> kernels(nr_kernels);
    size_t kernel_it = 0;
    for (kernel_it = 0; kernel_it < kernels.size(); kernel_it++) {
      auto &kernel = kernels[kernel_it];
      if (kernel.init(rowsA, rowsB, nr_dpus, rows_per_dpu) == false) {
        break;
      }

      kernel.set_params(alpha, beta, false);
      kernel.set_A(A, true);
    }
    kernels.resize(kernel_it);

    show_trace("Running {} kernels. Each kernel with {} DPUs.\n", kernels.size(), nr_dpus);

    size_t cur_kernel = 0;
    for (uint32_t i = 0; i < colsB; i++) {
      auto &kernel = get_free_kernel(kernels, cur_kernel);
      kernel.set_x(B + rowsB * i, true);
      kernel.set_y(C + rowsA * i, true);
      kernel.launch(true);
      kernel.get_y(C + rowsA * i, true);
    }

    for (auto &kernel : kernels) {
      kernel.sync();
      kernel.free_dpus();
    }
  }
}
}