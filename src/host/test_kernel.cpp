#include "common.hpp"
#include "dpu_transfer_helper.hpp"
#include "kernel.hpp"
#include "pimblas.h"

struct perfcounter_pair {
  uint64_t cycles;
  uint64_t instr;
};

perfcounter_pair get_max(perfcounter_pair *arr, size_t len) {
  perfcounter_pair ret{};
  for (size_t i = 0; i < len; i++) {
    if (arr[i].cycles > ret.cycles) {
      ret.cycles = arr[i].cycles;
    }
    if (arr[i].instr > ret.instr) {
      ret.instr = arr[i].instr;
    }
  }
  return ret;
}

void test_arithmetic() {
  struct params {
    uint32_t no_tasklets;
    uint32_t num_iters;
    int int_val;
    float float_val;
  };
  uint32_t nr_dpus = 1;

  dpu_set_t dpu_set;
  DPU_ASSERT(dpu_alloc(nr_dpus, nullptr, &dpu_set));

  Kernel kernel;
  kernel.set_dpu_set(dpu_set, nr_dpus);

  kernel.load_program("test_arithmetic.kernel");

  std::vector<params> argsList{
      {.no_tasklets = 16, .num_iters = 128, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 16, .num_iters = 256, .int_val = 7, .float_val = 7.0f},
      {.no_tasklets = 16, .num_iters = 512, .int_val = 7, .float_val = 7.0f},
      {.no_tasklets = 4, .num_iters = 128, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 8, .num_iters = 128, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 12, .num_iters = 128, .int_val = 3, .float_val = 3.0f},
  };

  for (auto &args : argsList) {
    kernel.set_arg_broadcast_exact("args", 0, &args, sizeof(args), false);

    kernel.launch(false);

    dpu_set_t dpu;
    uint32_t dpu_idx;
    DPU_FOREACH(dpu_set, dpu, dpu_idx) {
      constexpr size_t MAX_NR_TASKLETS = 16;

      std::cout << "DPU no. " << dpu_idx << "\n";

      auto read_and_print_res = [&dpu, &args](const char *sym_name) {
        perfcounter_pair results[MAX_NR_TASKLETS];
        copy_full(dpu, sym_name, 0, results, MAX_NR_TASKLETS);
        auto res = get_max(results, MAX_NR_TASKLETS);
        std::cout << args.no_tasklets << ", "
                  << args.num_iters << ", "
                  << args.int_val << ", "
                  << args.float_val << ", "
                  << sym_name << ", " 
                  << res.cycles << ", " 
                  << res.instr << "\n";
      };

      read_and_print_res("addition_int_results");
      read_and_print_res("multiplication_int_results");
      read_and_print_res("division_int_results");
      read_and_print_res("addition_float_results");
      read_and_print_res("multiplication_float_results");
      read_and_print_res("division_float_results");
    }
  }
  kernel.free_dpus();
}

void test_wram() {
  struct params {
    uint32_t no_tasklets;
    uint32_t num_iters;
    int int_val;
    float float_val;
  };
  uint32_t nr_dpus = 1;

  dpu_set_t dpu_set;
  DPU_ASSERT(dpu_alloc(nr_dpus, nullptr, &dpu_set));

  Kernel kernel;
  kernel.set_dpu_set(dpu_set, nr_dpus);

  kernel.load_program("test_wram_access.kernel");

  std::vector<params> argsList{
      {.no_tasklets = 16, .num_iters = 128, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 16, .num_iters = 256, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 16, .num_iters = 512, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 12, .num_iters = 256, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 10, .num_iters = 256, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 8, .num_iters = 256, .int_val = 3, .float_val = 3.0f},
      {.no_tasklets = 6, .num_iters = 256, .int_val = 3, .float_val = 3.0f},
  };

  for (auto &args : argsList) {
    kernel.set_arg_broadcast_exact("args", 0, &args, sizeof(args), false);

    kernel.launch(false);

    dpu_set_t dpu;
    uint32_t dpu_idx;
    DPU_FOREACH(dpu_set, dpu, dpu_idx) {
      constexpr size_t MAX_NR_TASKLETS = 16;

      std::cout << "DPU no. " << dpu_idx << "\n";

      auto read_and_print_res = [&dpu, &args](const char *sym_name) {
        perfcounter_pair results[MAX_NR_TASKLETS];
        copy_full(dpu, sym_name, 0, results, MAX_NR_TASKLETS);
        auto res = get_max(results, MAX_NR_TASKLETS);
        std::cout << args.no_tasklets << ", "
                  << args.num_iters << ", "
                  << args.int_val << ", "
                  << args.float_val << ", "
                  << sym_name << ", " 
                  << res.cycles << ", " 
                  << res.instr << "\n";
      };

      read_and_print_res("read_and_add_int_results");
      read_and_print_res("write_and_add_int_results");
      read_and_print_res("rw_and_add_int_results");
      read_and_print_res("read_and_add_float_results");
      read_and_print_res("write_and_add_float_results");
      read_and_print_res("rw_and_add_float_results");
    }
  }
  kernel.free_dpus();
}

void test_mram() {
  struct params {
    uint32_t no_tasklets;
    uint32_t num_elems_mram;
    uint32_t num_elems_wram_per_tasklet;
    int step;
  };
  uint32_t nr_dpus = 1;

  dpu_set_t dpu_set;
  DPU_ASSERT(dpu_alloc(nr_dpus, nullptr, &dpu_set));

  Kernel kernel;
  kernel.set_dpu_set(dpu_set, nr_dpus);

  kernel.load_program("test_mram.kernel");

  std::vector<params> argsList{
      // 40KB of WRAM is the maximum I could allocate
      {.no_tasklets = 1, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 1024 * 10, .step = 1},
      {.no_tasklets = 4, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 1024 * 10 / 4, .step = 1},
      {.no_tasklets = 8, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 1024 * 10 / 8, .step = 1},
      {.no_tasklets = 12, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 1024 * 10 / 12, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 1024 * 10 / 16, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 16, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 32, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 64, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 128, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 1},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 16, .step = 2},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 32, .step = 2},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 64, .step = 2},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 128, .step = 2},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 2},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 2},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 16, .step = 4},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 32, .step = 4},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 64, .step = 4},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 128, .step = 4},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 4},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 4},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 16, .step = 8},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 32, .step = 8},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 64, .step = 8},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 128, .step = 8},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 8},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 8},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 32, .step = 16},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 64, .step = 16},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 128, .step = 16},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 16},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 16},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 64, .step = 32},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 128, .step = 32},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 32},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 32},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 128, .step = 64},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 64},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 64},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 256, .step = 128},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 128},
      {.no_tasklets = 16, .num_elems_mram = 1024 * 1024 * 16, .num_elems_wram_per_tasklet = 512, .step = 256},
  };

  for (auto &args : argsList) {
    kernel.set_arg_broadcast_exact("args", 0, &args, sizeof(args), false);

    kernel.launch(false);
    dpu_set_t dpu;
    uint32_t dpu_idx;
    DPU_FOREACH(dpu_set, dpu, dpu_idx) {
      constexpr size_t MAX_NR_TASKLETS = 16;

      auto read_and_print_res = [&dpu, &args](const char *sym_name) {
        perfcounter_pair results[MAX_NR_TASKLETS];
        copy_full(dpu, sym_name, 0, results, MAX_NR_TASKLETS);
        auto res = get_max(results, MAX_NR_TASKLETS);
        std::cout << args.no_tasklets << ", "
                  << args.num_elems_mram * 4 << ", "
                  << args.num_elems_wram_per_tasklet * 4 << ", "
                  << args.step << ", "
                  <<  sym_name << ", " 
                  << res.cycles << ", " 
                  << res.instr << "\n";
      };

      read_and_print_res("seq_read_and_add_int_results");
      read_and_print_res("seq_rw_and_add_int_results");
      read_and_print_res("seq_read_and_mul_int_results");
      read_and_print_res("seq_rw_and_mul_int_results");
      read_and_print_res("seq_read_and_div_int_results");
      read_and_print_res("seq_rw_and_div_int_results");
      read_and_print_res("seq_read_and_add_float_results");
      read_and_print_res("seq_rw_and_add_float_results");
      read_and_print_res("seq_read_and_mul_float_results");
      read_and_print_res("seq_rw_and_mul_float_results");
      read_and_print_res("seq_read_and_div_float_results");
      read_and_print_res("seq_rw_and_div_float_results");
    }
  }
  kernel.free_dpus();
}

extern "C" {
void test_kernel() {
  test_arithmetic();

  //test_wram();

  //test_mram();
}
}