#pragma once
#include "kernel.hpp"

class GEMVInt32_Kernel_Beta : public Kernel {
  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    int alpha;
    int beta;
  };

 public:
  void set_A(const int *data, bool async);

  void set_x(const int *data, bool async);

  void set_y(const int *data, bool async);

  void get_y(int *data, bool async);

  void set_params(const int *alpha, const int *beta, bool async);

  void init(uint32_t m, uint32_t n);
  bool init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu);

  bool running = false;

 private:
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};

class GEMVInt32_Kernel : public Kernel {
  struct params {
    uint32_t rows_per_dpu;
    uint32_t row_size;
    int alpha;
  };

 public:
  void set_A(const int *data, bool async);

  void set_x(const int *data, bool async);

  void get_y(int *data, bool async);

  void set_params(const int *alpha, bool async);

  void init(uint32_t m, uint32_t n);
  bool init(uint32_t m, uint32_t n, uint32_t nr_dpus, uint32_t rows_per_dpu);

  bool running = false;

 private:
  uint32_t m;
  uint32_t n;
  uint32_t rows_per_dpu;

  size_t A_offset;
  size_t x_offset;
  size_t y_offset;
};