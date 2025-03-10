#include <iomanip>
#include <limits>

#include "common.hpp"
#include "test_helper.hpp"

std::vector<float> create_sample_data(size_t size) {
  std::vector<float> buffer(size, 0);
  for (size_t i = 0; i < size; i++) {
    buffer[i] = (float)i - ((float)size / 2);
  }
  return buffer;
}

void host_add(float *input_a, float *input_b, float *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] + input_b[i];
  }
}

void host_mul(float *input_a, float *input_b, float *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] * input_b[i];
  }
}

void host_sub(float *input_a, float *input_b, float *output, size_t size) {
  for (size_t i = 0; i < size; i++) {
    output[i] = input_a[i] - input_b[i];
  }
}

int single_test_add(size_t size) {
  auto sample_data_a = create_sample_data(size);
  auto sample_data_b = create_sample_data(size);

  float *result = new float[size];
  vec_add_f(sample_data_a.data(), sample_data_b.data(), result, size);

  float *result_host = new float[size];
  host_add(sample_data_a.data(), sample_data_b.data(), result_host, size);

  for (size_t i = 0; i < size; i++) {
    if (abs(result[i] - result_host[i]) > 0.01) {
      return -1;
    }
  }
  return 0;
}

int single_test_mul(size_t size) {
  auto sample_data_a = create_sample_data(size);
  auto sample_data_b = create_sample_data(size);

  float *result = new float[size];
  vec_mul_f(sample_data_a.data(), sample_data_b.data(), result, size);

  float *result_host = new float[size];
  host_mul(sample_data_a.data(), sample_data_b.data(), result_host, size);

  for (size_t i = 0; i < size; i++) {
    if (abs(result[i] - result_host[i]) > 0.01) {
      return -1;
    }
  }
  return 0;
}

int single_test_sub(size_t size) {
  auto sample_data_a = create_sample_data(size);
  auto sample_data_b = create_sample_data(size);

  float *result = new float[size];
  vec_sub_f(sample_data_a.data(), sample_data_b.data(), result, size);

  float *result_host = new float[size];
  host_sub(sample_data_a.data(), sample_data_b.data(), result_host, size);

  for (size_t i = 0; i < size; i++) {
    if (abs(result[i] - result_host[i]) > 0.01) {
      return -1;
    }
  }
  return 0;
}

int main(int argc, char **argv) {
  for (int i = 0; i < 10; i++) {
    if (single_test_add(10000 + i) != 0) {
      printf("ADD TEST FAILED\n");
      RET_TEST_FAIL;
    }
  }
  for (int i = 0; i < 10; i++) {
    if (single_test_mul(10000 + i) != 0) {
      printf("MUL TEST FAILED\n");
      RET_TEST_FAIL;
    }
  }
  for (int i = 0; i < 10; i++) {
    if (single_test_sub(10000 + i) != 0) {
      printf("SUB TEST FAILED\n");
      RET_TEST_FAIL;
    }
  }
  printf("TEST OK\n");
  RET_TEST_OK;
}
