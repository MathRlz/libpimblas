#include <cstdint>
#include <memory>

#include "gemvf_kernel.hpp"

class dpu_mat {
 public:
  dpu_mat();
  ~dpu_mat();

  int initialize(const float *matrix, uint32_t rows, uint32_t cols);
  int gemv(const float *vecIn, float *vecOut);

 private:
  std::unique_ptr<GEMVF_Kernel> kernel;
};