#include <chrono>

#include "common.hpp"
#include "test_helper.hpp"

class Timer {
 public:
  void start() { _start = std::chrono::high_resolution_clock::now(); }
  void finish() { _finish = std::chrono::high_resolution_clock::now(); }
  size_t timeInMs() { return std::chrono::duration_cast<std::chrono::milliseconds>(_finish - _start).count(); }

  size_t timeInNs() { return std::chrono::duration_cast<std::chrono::nanoseconds>(_finish - _start).count(); }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> _start;
  std::chrono::time_point<std::chrono::high_resolution_clock> _finish;
};

int main() {
  Timer timer;

  dpu_mat_handle_t mat_handle;
  if (create_context(&mat_handle) != 0) {
    RET_TEST_FAIL;
  }

  uint32_t m = 8192;
  uint32_t n = 8192;
  float floatMin = -10.0f;
  float floatMax = 10.0f;
  // row major
  auto matrix = generateRandomFloats(m * n, floatMin, floatMax);

  timer.start();
  if (initialize_context(mat_handle, matrix.data(), m, n) != 0) {
    RET_TEST_FAIL;
  }
  timer.finish();
  std::cout << "Initialization took " << timer.timeInNs() << "ns.\n";

  uint32_t numVecsToMul = 1024;
  pimblas::vector<float> vecOut(n);
  for (uint32_t i = 0; i < numVecsToMul; i++) {
    auto vec = generateRandomFloats(n, floatMin, floatMax);
    timer.start();
    if (multiply_by_vector(mat_handle, vec.data(), vecOut.data()) != 0) {
      RET_TEST_FAIL;
    }
    timer.finish();
    std::cout << "Mul (" << i << ") took " << timer.timeInNs() << "ns.\n";
  }

  if (destroy_context(mat_handle) != 0) {
    RET_TEST_FAIL;
  }

  RET_TEST_OK;
}
