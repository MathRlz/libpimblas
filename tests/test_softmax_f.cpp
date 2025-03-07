#include "common.hpp"
#include "test_helper.hpp"

int main() {
    size_t vec_size = 214378;
    pimblas::vector<float> vec = generateRandomFloats(vec_size, 0.0f, 10000.0f);
    pimblas::vector<float> vec_softmax(vec_size, 0.0f);
    vec[vec_size-1] = 100000000.0f;
    if (softmax(vec.data(), vec_softmax.data(), vec.size()) != 0){
        RET_TEST_FAIL;
    }

    std::cout << vec_softmax[0] << " " << vec_softmax[vec_size-1] << std::endl;

    float sum=0.0f;
    for (auto val : vec_softmax) {
        sum+=val;
    }
    std::cout << "it sums up to: " << sum << std::endl;

    RET_TEST_OK;
}