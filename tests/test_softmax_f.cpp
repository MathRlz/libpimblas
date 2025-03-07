#include "common.hpp"
#include "test_helper.hpp"

int main() {

    size_t vec_size = 742;
    pimblas::vector<float> vec(vec_size, 0.0f);
    pimblas::vector<float> vec_softmax(vec_size, 0.0f);
    vec[31] = 14.0f;
    if (softmax(vec.data(), vec_softmax.data(), vec.size()) != 0){
        RET_TEST_FAIL;
    }

    std::cout << vec_softmax[0] << " " << vec_softmax[31] << std::endl;

    float sum=0.0f;
    for (auto val : vec_softmax) {
        sum+=val;
    }
    std::cout << "it sums up to: " << sum << std::endl;

    RET_TEST_OK;
}