#include <chrono>
#include <fstream>

#include "common.hpp"
#include "dpu_transfer_helper.hpp"
#include "kernel.hpp"
#include "test_helper.hpp"

int host_gemv_int8(uint32_t m, uint32_t n, const int8_t *mat, const int8_t *vec, int *y, int alpha, int beta) {
  for (size_t row = 0; row < m; ++row) {
    int mul_res = 0;
    for (size_t col = 0; col < n; ++col) {
      mul_res += static_cast<int>(vec[col]) * static_cast<int>(mat[row * n + col]);
    }
    y[row] = alpha * mul_res + y[row] * beta;
  }

  return 0;
}

struct params {
  uint32_t rows_per_dpu;
  uint32_t row_size;
  int32_t alpha;
  int32_t beta;
};

struct BenchResult {
  long mat_transfer_ns = 0;
  long vec_transfer_ns = 0;
  long y_in_transfer_ns = 0;
  long y_out_transfer_ns = 0;
  long compute_ns = 0;
};

struct Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;

  void start_timer() { start = std::chrono::high_resolution_clock::now(); }

  long stop_timer() {
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
};

BenchResult launch_gemv_int8_opt(uint32_t M, uint32_t N, uint32_t nr_dpus, int alpha, int beta, int8_t *mat,
                                 int8_t *vec, int *out) {
  constexpr uint32_t min_rows_per_dpu = 16 * 8 / sizeof(int);
  uint32_t rows_per_dpu = alignUp((M - 1) / nr_dpus + 1, min_rows_per_dpu);
  Kernel k;
  if (false == k.allocate_n(nr_dpus)) {
    std::cerr << "Error while allocating\n";
  }
  k.load_program("gemv_int8.kernel");

  params args{.rows_per_dpu = rows_per_dpu, .row_size = N, .alpha = alpha, .beta = beta};
  k.set_arg_broadcast_exact("args", 0, reinterpret_cast<uint8_t *>(&args), sizeof(params), false);
  uint32_t x_offset = alignUp(rows_per_dpu * N * sizeof(int8_t), 8);
  uint32_t y_offset = x_offset + alignUp(N * sizeof(int8_t), 8);

  BenchResult results;
  Timer t;
  t.start_timer();
  // std::cout << "Mat.\n" << std::endl;
  k.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, 0, mat, rows_per_dpu * N * sizeof(int8_t), M * N * sizeof(int8_t), false);
  results.mat_transfer_ns = t.stop_timer();

  t.start_timer();
  // std::cout << "Vec.\n" << std::endl;
  k.set_arg_broadcast(DPU_MRAM_HEAP_POINTER_NAME, x_offset, vec, N * sizeof(int8_t), false);
  results.vec_transfer_ns = t.stop_timer();

  if (beta != 0) {
    t.start_timer();
    k.set_arg_scatter(DPU_MRAM_HEAP_POINTER_NAME, y_offset, out, rows_per_dpu * sizeof(int), M * sizeof(int), false);
    results.y_in_transfer_ns = t.stop_timer();
  }

  t.start_timer();
  /*
  if (false == k.var_launch(false)) {
    std::cerr << "error while launching kernel.\n";
    return {};
  }
    */
  k.launch(false);
  results.compute_ns = t.stop_timer();

  t.start_timer();
  // std::cout << "result.\n" << std::endl;
  k.get_arg_gather_safe(DPU_MRAM_HEAP_POINTER_NAME, y_offset, out, rows_per_dpu * sizeof(int), M * sizeof(int));
  results.y_out_transfer_ns = t.stop_timer();
  return results;
}

struct BenchRun {
  uint32_t M;
  uint32_t N;
  uint32_t nr_dpus;
  int alpha;
  int beta;
  int min_val;
  int max_val;
};

std::vector<std::tuple<BenchResult, BenchRun>> gemv_int8_opt_test(const std::vector<BenchRun> &runs, int probes = 10) {
  std::vector<std::tuple<BenchResult, BenchRun>> results;
  for (const auto &run : runs) {
    std::cout << "Running M: " << run.M << ", N: " << run.N << ", nr_dpus: " << run.nr_dpus << ", alpha: " << run.alpha
              << ", beta: " << run.beta << std::endl;

    std::vector<BenchResult> bench_results;
    for (int i = 0; i < probes; i++) {
      auto mat = generateRandomIntegral<int8_t>(static_cast<size_t>(run.M) * static_cast<size_t>(run.N), run.min_val,
                                                run.max_val);
      auto vec = generateRandomIntegral<int8_t>(run.N, run.min_val, run.max_val);
      auto out = generateRandomIntegers(run.M, -100, 100);
      try {
        auto result =
            launch_gemv_int8_opt(run.M, run.N, run.nr_dpus, run.alpha, run.beta, mat.data(), vec.data(), out.data());
        bench_results.push_back(result);
      } catch (const std::exception &e) {
        std::cerr << "Error in kernel execution: " << e.what() << std::endl;
      }
    }
    BenchResult final_result;
    final_result.mat_transfer_ns =
        std::accumulate(bench_results.begin(), bench_results.end(), 0L,
                        [](long sum, const BenchResult &br) { return sum + br.mat_transfer_ns; }) /
        probes;
    final_result.vec_transfer_ns =
        std::accumulate(bench_results.begin(), bench_results.end(), 0L,
                        [](long sum, const BenchResult &br) { return sum + br.vec_transfer_ns; }) /
        probes;
    final_result.y_in_transfer_ns =
        std::accumulate(bench_results.begin(), bench_results.end(), 0L,
                        [](long sum, const BenchResult &br) { return sum + br.y_in_transfer_ns; }) /
        probes;
    final_result.y_out_transfer_ns =
        std::accumulate(bench_results.begin(), bench_results.end(), 0L,
                        [](long sum, const BenchResult &br) { return sum + br.y_out_transfer_ns; }) /
        probes;
    final_result.compute_ns = std::accumulate(bench_results.begin(), bench_results.end(), 0L,
                                              [](long sum, const BenchResult &br) { return sum + br.compute_ns; }) /
                              probes;

    results.push_back(std::make_tuple(final_result, run));
  }
  return results;
}

void write_to_csv(const std::vector<std::tuple<BenchResult, BenchRun>> &results, const std::string &out_filename) {
  auto out_file = std::ofstream(out_filename);
  if (false == out_file.is_open()) {
    std::cerr << "Couldn't create " << out_filename << std::endl;
  }

  out_file << "M,N,nr_dpus,mat_transfer_ns,vec_transfer_ns,y_in_transfer_ns,y_out_transfer_ns,compute_ns\n";
  for (const auto &[result, run] : results) {
    out_file << run.M << "," << run.N << "," << run.nr_dpus << "," << result.mat_transfer_ns << ","
             << result.vec_transfer_ns << "," << result.y_in_transfer_ns << "," << result.y_out_transfer_ns << ","
             << result.compute_ns << "\n";
  }
}

int main(int argc, char **argv) {
  std::vector<BenchRun> runs;

  // Single dpu configurations
  if (false) {
    for (uint32_t m = 1; m <= 1024; m *= 2) {
      auto rows_per_dpu = alignUp(m, 32);
      for (uint32_t n = 1; n < 64 * 1024 * 1024; n *= 2) {
        size_t dpu_mem_req = alignUp(rows_per_dpu * n * sizeof(int8_t), 8) + alignUp(n * sizeof(int8_t), 8) +
                             alignUp(rows_per_dpu * sizeof(int), 8);
        if (dpu_mem_req >= (32 * 1024 * 1024)) {
          break;
        }
        runs.push_back({m, n, 1, 1, 0, -128, 127});
      }
    }

    // Kernel only works up to 1696 rows per dpu
    for (uint32_t m = 1024; m <= 1696; m += 32) {
      auto rows_per_dpu = alignUp(m, 32);
      for (uint32_t n = 1; n < 64 * 1024 * 1024; n *= 2) {
        size_t dpu_mem_req = alignUp(rows_per_dpu * n * sizeof(int8_t), 8) + alignUp(n * sizeof(int8_t), 8) +
                             alignUp(rows_per_dpu * sizeof(int), 8);
        if (dpu_mem_req >= (63 * 1024 * 1024)) {
          break;
        }
        runs.push_back({m, n, 1, 1, 0, -128, 127});
      }
    }
  }

  std::vector<uint32_t> nr_dpus_conf = {1, 2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024, 2048, 2551};
  for (auto nr_dpus : nr_dpus_conf) {
    uint32_t start_m = (nr_dpus > 1) ? 32 : 1;
    for (uint32_t m = start_m; m <= 1024; m *= 2) {
      uint32_t M = m * nr_dpus;
      auto rows_per_dpu = alignUp((M - 1) / nr_dpus + 1, 32);

      if (M < (nr_dpus - 1) * rows_per_dpu) {
        continue;
      }

      for (uint32_t n = 1; n <= 16 * 1024; n *= 2) {
        uint32_t N = n;
        size_t dpu_mem_req = alignUp(rows_per_dpu * N * sizeof(int8_t), 8) + alignUp(N * sizeof(int8_t), 8) +
                             alignUp(rows_per_dpu * sizeof(int), 8);
        if (dpu_mem_req >= (63 * 1024 * 1024)) {
          break;
        }
        size_t sys_mem_req =
            alignUp(M * N * sizeof(int8_t), 8) + alignUp(N * sizeof(int8_t), 8) + alignUp(M * sizeof(int), 8);
        constexpr size_t sys_mem_thresh = 64LU * 1024LU * 1024LU * 1024LU;  // 64GB
        if (sys_mem_req >= sys_mem_thresh) {
          std::cout << "sys_mem_req:" << sys_mem_req << std::endl;
          break;
        }
        runs.push_back({M, N, nr_dpus, 1, 0, -128, 127});
      }
    }
  }

  auto results = gemv_int8_opt_test(runs);
  for (const auto &[result, run] : results) {
    std::cout << "M: " << run.M << ", N: " << run.N << ", nr_dpus: " << run.nr_dpus << ", alpha: " << run.alpha
              << ", beta: " << run.beta << ", min_val: " << run.min_val << ", max_val: " << run.max_val << std::endl;
    std::cout << "Matrix Transfer: " << static_cast<float>(result.mat_transfer_ns) / 1000000 << " ms, "
              << "Vector Transfer: " << static_cast<float>(result.vec_transfer_ns) / 1000000 << " ms, "
              << "Y Input Transfer: " << static_cast<float>(result.y_in_transfer_ns) / 1000000 << " ms, "
              << "Y Output Transfer: " << static_cast<float>(result.y_out_transfer_ns) / 1000000 << " ms, "
              << "Compute Time: " << static_cast<float>(result.compute_ns) / 1000000 << " ms" << std::endl;
    std::cout << std::endl;
  }
  write_to_csv(results, "gemv_int8_opt_results.csv");
}
