#include "common.hpp"
#include "test_helper.hpp"

void softmax_host(const pimblas::vector<float> &vec, pimblas::vector<float> &out_softmax) {
  float max = *std::max_element(vec.begin(), vec.end());
  float sum = 0.0f;
  for (size_t i = 0; i < vec.size(); i++) {
    out_softmax[i] = expf(vec[i] - max);
    sum += out_softmax[i];
  }
  for (size_t i = 0; i < vec.size(); i++) {
    out_softmax[i] /= sum;
  }
}

int main() {
  size_t vec_size = 10244317;
  pimblas::vector<float> vec = generateRandomFloats(vec_size, 0.0f, 100.0f);
  pimblas::vector<float> vec_softmax(vec_size, 0.0f);

  if (softmax(vec.data(), vec_softmax.data(), vec.size()) != 0) {
    RET_TEST_FAIL;
  }

  pimblas::vector<float> vec_softmax_host(vec_size, 0.0f);
  softmax_host(vec, vec_softmax_host);

  float tolerance = 1e-6f;
  bool same = mostly_same_abs(vec_softmax.data(), vec_softmax_host.data(), vec_size, tolerance);
  if (false == same) {
    RET_TEST_FAIL;
  }

  std::cout << "SUCCESS" << std::endl;
  RET_TEST_OK;
}