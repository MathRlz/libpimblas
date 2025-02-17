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
    size_t size = 32 * 1024 * 1024;
    auto vec1 = generateRandomIntegers<uint32_t>(size, std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());
    auto vec2 = generateRandomIntegers<uint32_t>(size, std::numeric_limits<uint32_t>::min(), std::numeric_limits<uint32_t>::max());

    auto host_result = dotProduct(vec1.data(), vec2.data(), vec1.size());
    uint64_t dpu_result;
    if (clever_dot_product(vec1.data(), vec2.data(), vec1.size(), &dpu_result) != 0) {
        RET_TEST_FAIL;
    }

    if (host_result != dpu_result) {
        std::cout << "Results differ. Host_result=" << host_result << ", dpu_result=" << dpu_result << std::endl;
        RET_TEST_FAIL;
    }

    std::cout << "SUCCESS\n";
    RET_TEST_OK;
}