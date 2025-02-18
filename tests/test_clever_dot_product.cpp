#include "common.hpp"
#include "test_helper.hpp"

uint64_t dotProduct(const uint32_t *vec1, const uint32_t *vec2, size_t size) { 
    uint64_t result = 0;
    for (size_t i = 0; i < size; i++) {
        result += static_cast<uint64_t>(vec1[i]) * static_cast<uint64_t>(vec2[i]);
    }
    return result;
}

int main() {
    size_t size = 1024;
    auto vec1 = generateRandomIntegers<uint32_t>(size, std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
    auto vec2 = generateRandomIntegers<uint32_t>(size, std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());

    auto host_result = dotProduct(vec1.data(), vec2.data(), vec1.size());
    uint64_t clever_dpu_result;
    if (clever_dot_product(vec1.data(), vec2.data(), vec1.size(), &clever_dpu_result) != 0) {
        RET_TEST_FAIL;
    }

    if (host_result != clever_dpu_result) {
        std::cout << "Results differ. Host_result=" << host_result << ", dpu_result=" << clever_dpu_result << std::endl;
        RET_TEST_FAIL;
    }

    uint64_t simple_dpu_result;
    if (dot_product(vec1.data(), vec2.data(), vec1.size(), &simple_dpu_result) != 0) {
        RET_TEST_FAIL;
    }

    if (host_result != simple_dpu_result) {
        std::cout << "Results differ. Host result= " << host_result << ", simple dot dpu result= " << simple_dpu_result << std::endl;
        RET_TEST_FAIL;
    }

    std::cout << "SUCCESS\n";
    RET_TEST_OK;
}