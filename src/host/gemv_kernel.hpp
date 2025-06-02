#pragma once
#include "kernel.hpp"

template <typename inType, typename outType>
class GEMV_Kernel : public Kernel {
  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    outType alpha;
    outType beta;
  };

 public:
  GEMV_Kernel() = delete;
  GEMV_Kernel(const std::string &program_name) : program_name(program_name) {}

  void set_A(const inType *data, bool async);

  void set_x(const inType *data, bool async);

  void set_y(const outType *data, bool async);

  void get_y(outType *data, bool async);

  void get_y_safe(outType *data);

  void set_params(const outType *alpha, const outType *beta, bool async);

  bool init(uint32_t m, uint32_t n);
  bool init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu);

 private:
  std::string program_name;
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};

#include "gemv_kernel_impl.tpp"

class GEMVF_Kernel : public GEMV_Kernel<float, float> {
 public:
  GEMVF_Kernel() : GEMV_Kernel("gemv_f.kernel") {}
};

class GEMV_INT8_Kernel : public GEMV_Kernel<int8_t, int> {
 public:
  GEMV_INT8_Kernel() : GEMV_Kernel("gemv_int8.kernel") {}
};

class GEMV_INT32_Kernel : public GEMV_Kernel<int, int> {
 public:
  GEMV_INT32_Kernel() : GEMV_Kernel("gemv_int32.kernel") {}
};
